# Task 5 plan: sample-in-graph / terminal-sync removal (handoff) — SERVER PATH ONLY

> **OUTCOME: STOP (2026-10-03).** Phase 0 measured the prize and it is not
> there: the server step is **device-bound with zero slack**, and deleting
> the terminal sync outright -- an illegal change, the plan's own ceiling --
> buys only **0.283 ms/step (+1.63%)**, reachable only by the "LAST,
> hardest" attack the plan itself gates behind a static-Philox RNG spike.
> No production code was written. Full bill in **§7**, ledger entry in
> `doc/kvarn-4090.md` ("Sampler + terminal sync — STOP").

**Premise warning (read first): this cut likely does NOT move the
ledger tg number.** Ledger tg is `eval/kvarn_microkld.py` greedy
decode (`argmax`, no `.item()`, no terminal sync). The syncs below
live in the SERVER generator loop
(`exllamav3/generator/generator.py`, `job.py`). If the goal metric
stays microkld-tg, SKIP after §2 and record the skip. If server
tokens/s matters, proceed.

Branch: `wip/kvarn-cache` (tip, do NOT rebase). Scope: generator
loop terminal syncs only, 4090 box. No sampler-behavior change
(same tokens out for same logits in — twin is token-identity).

## 0. Facts (in-code, verify before touching)

- `generator.py:670`: greedy path `torch.argmax` (GPU-side, no
  sync by itself).
- `generator.py:1128`: `torch.cuda.synchronize(device)` — FULL
  device sync in the loop. Find what it guards (timing? logits
  handoff?) — that guard dictates the replacement.
- `generator.py:677,783`: `c.view(-1).tolist()` per step
  (calibration estimates); `:889` `cuts.max().item()`;
  `:1222` per-token `.item()` compares (draft path);
  `:1282-1283` `.item()` recalibration reads.
- `job.py:621`: `next_token.item()` — the per-step token D2H.
- Sampler object: `DefaultSampler` (`presets.py:3`, not `job.py:166`);
  greedy is `ArgmaxSampler` = `SS_Argmax` (`presets.py:18`).
  **STATUS: facts verified, two corrections.** `generator.py:670` is the
  *draft* path, not the main loop; the main loop's greedy sample is
  `job.receive_logits` → `self.sampler.forward` (`job.py:571-583`). The
  sampler takes the RNG as a **host int** (`self.rng.randint(...)`,
  `job.py:578`) passed into `forward(..., rand_u32, ...)`
  (`sampler/custom.py:1171-1190`), which is the concrete form of the
  static-Philox risk — see §7.5.
- **The `.item()` batching this plan's §3.1 proposes ALREADY EXISTS**:
  `generator.py:1102-1128` stages every job's token into one pinned
  buffer and pays exactly ONE `torch.cuda.synchronize` per step, then
  collects. The plan's §0 read of `:1128` as "FULL device sync in the
  loop" is right about the call but misses that the per-job round trips
  were already collapsed. Measured cost of what remains: 4 reads/step,
  3.8us total (§7.2).

## 1. Read this first (no code until done)

**STATUS: DONE.** All four reads done. §7.2 is the resulting bill; the
corrections to §0's reading are listed there.

1. `generator.py:650-700` and `:770-800` (the two sample sites +
   surrounding orchestration: what consumes the token on host
   and when).
2. `generator.py:1110-1140` (the full synchronize: what breaks
   if it goes away).
3. `job.py:600-640` (per-step token extraction + stop checks).
4. Task-6 plan §8 (in-process 19.27ms vs server 21ms: ~1.7ms is
   loop overhead ABOVE the model — your prize pool ceiling).

## 2. Phase 0 — premise check + STOP gate (measure, no code)

**STATUS: RAN. Gate resolved by measurement: the raw sync (0.667 ms) is
above the 0.5 ms bar, but the RECOVERABLE prize is 0.283 ms — the step
is device-bound with zero slack, so the bar's purpose ("the prize is
below the bar and the RNG-in-graph risk is not worth it") applies.
STOP recorded in §7 and in the ledger. Nothing below §2 was started.**

1. Instrument ONE server-harness 64k run (tabbyAPI or a minimal
   `generator.py` loop, NOT microkld): CUDA-event time per step
   split into (model forward) vs (sample + item + sync +
   loop overhead). Write the bill down.
2. If terminal/sample/loop overhead <0.5ms/step: STOP, record
   skip note in ledger (2 lines), do not proceed — the prize is
   below the bar and the RNG-in-graph risk is not worth it.
3. If ≥0.5ms: rank the contributors (full-sync vs item-calls vs
   sampler kernel) and attack in that order, one at a time.

## 3. Ranked attacks (only after §2 bar passes)

**STATUS: NOT RUN — barred by §2 (recoverable prize 0.283 ms < 0.5 ms).
Ranking below was still scored against the measurement, because §2.3
asks for it: attack 1 is already shipped, attack 2 is worth ~0 for a
reason §7.4 gives, attack 3 is the only one with any prize and it is
the one carrying the RNG risk. No env gate
(`EXL3_KVARN_SERVER_TRIM`) was added.**

1. **Batch the `.item()` reads.** `next_token.item()` + stop-check
   reads per step → one D2H per step max (single-element tensor
   copy, read once). No behavior change, no graph needed.
2. **Kill/replace the full `synchronize`.** If it guards timing:
   move it out of the hot loop (time every Nth step). If it
   guards correctness (logits handoff across streams): replace
   with a stream-event wait (no host stall). Prove with the
   token-identity twin.
3. **Argmax/sample inside the model graph (LAST, hardest).** Only
   if 1+2 leave prize: capture final-norm → lm_head → argmax as
   a graph tail; host gets the token via one async copy. RNG
   paths (temp/top-k multinomial) need static Philox counters in
   the capture — spike FIRST (`eval/_spike11_sample.py`, never
   commit), greedy-only before stochastic. Stop strings / EOS
   checks stay on host (they need the token value, not the
   distribution).

## 4. Production rules + twin (binding)

**STATUS: NOT APPLICABLE — no cut to gate.** No twin was needed: there
is no code change to twin against, and the twin as written (token
identity over 256 greedy steps) could not have been run against a
"sample in graph" that was never written. The one thing §4's twin would
have protected — greedy token identity — is unchanged by construction,
since no sampler code was touched.

- Token-identity twin (no tolerance — sampling must be
  bit-identical for fixed seed / greedy): 256 greedy steps @8k,
  `torch.equal` on the full token sequence vs legacy loop, plus
  stop-string/EOS behavior spot-checks (3 canned prompts ending
  mid-generation).
- Env gate `EXL3_KVARN_SERVER_TRIM=1` (default OFF → ON after
  box-green), fail-closed to legacy loop, loud fallback print.
- Same §6 box protocol as task-6 plan, but measured in the
  SERVER harness (microkld cannot see this prize — do not
  validate here with microkld).

## 5. Non-goals + guards (binding, cf. task-6 plan §5/§5b)

**STATUS: HELD.** No sampler-math change, no stop-string change, no
prefill edit, no dispatch/kvarn change, no production file touched at
all. `git status`/`git diff --stat` checked before the (docs-only)
commit; every add explicit; spikes (`eval/_spike11_sampler_bill.py`,
`eval/_spike11.bat`) left untracked.

No sampler-math changes (same distribution, same seed ->
same tokens), no stop-string behavior change, no prefill edits.
Hands-off everything outside scope; `git status` + `git diff
--stat` before every commit, explicit per-file `git add`.
Spikes untracked, atomic commits, ledger lines on landing.

## 6. Done means

**STATUS: MET via the second branch — "documented SKIP with the §2
bill".** Server-harness tok/s at 64k ctx (greedy) is 57.66 tok/s
(17.344 ms/step) and the measured prize for the whole plan is +1.63%
at an absolute ceiling that requires an incorrect change; the bill is
§7. The twin / PARITY / env-gate gates in §4 gate a cut, and there was
no cut. Microkld tg was not chased — as the premise warning predicted,
this work cannot move it.

A note the premise warning got right and is worth keeping: the ledger
metric is microkld (greedy, in-process argmax loop), so it cannot see
this path at all. Per the warning's own instruction — "if the goal
metric stays microkld-tg, SKIP after §2 and record the skip" — that
alone closes this task.

Server-harness tok/s at 64k ctx (greedy) improves by the §2
measured prize (±20%) with token-identity twin + PARITY green,
ledger entry, pushed — or documented SKIP with the §2 bill.
Microkld tg is not expected to move; do not chase it here.

## 7. RESULTS (2026-10-03, box mirror `wip/kvarn-cache`, protocol v3,
##    spike `eval/_spike11_sampler_bill.py`, log `t5_bill.log`) — STOP.
##    Step is device-bound; terminal-sync ceiling is +1.63%.

No production code written. §3 was never started.

### 7.1 Harness (server path, not microkld)

Real `Generator` + `Job` (`ArgmaxSampler`), one 65536-token prompt,
`EXL3_KVARN_TRITON=1 EXL3_KVARN_IMAGELESS=1 EXL3_KVARN_TRITON_PARITY=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `kvarn4`,
Qwen3.8-27B 1.40bpw, RTX 4090). All instrumentation is monkeypatched
onto the live instances plus `torch.cuda.synchronize` and
`torch.Tensor.{item,cpu,tolist,nonzero}`; every wait and every readback
is attributed to a call site, so no blocking can hide in an unattributed
bucket. 6 windows × 40 steps per arm, interleaved, arm order flipped
every window.

### 7.2 The bill (median of 6 windows × 40 steps, base arm)

```
step wall          17.344 ms   57.66 tok/s
  model forward    16.382 ms   94.4% of the step   (host dispatch 11.72
                                           + blocked store reads 4.66)
  terminal sync     0.667 ms    3.9%   generator.py:1128, 1 per step
  everything else    0.245 ms    1.4%   staging, sampler launch,
                                         receive_sample, requeue/dequeue
    of which sampler launch (t_launch)   0.000 ms
    of which receive_sample + its 4 token readbacks  0.004 ms
```

Sampler device tail (`ev_fwd` → `ev_end`): **0.024 ms**.

`Tensor` readbacks per step, by call site (base arm, 40 steps):

| calls/step | ms/step | site |
|---|---|---|
| 16 | **4.629** | `kvarn_triton.py:502 tolist` — fused-store status readback, via `kvarn.py:2019 _store_rows` ← `kvarn.py:2986 update_kv_direct` ← `dispatch.py:259 _try_kvarn_graph_decode` ← `dispatch.py:476 _try_kvarn_online_decode` |
| 1 | 0.0019 | `job.py:620 .cpu` (batched path: already a pinned CPU tensor) |
| 1 | 0.0016 | `job.py:621 .item` |
| 1 | 0.0005 | `job.py:809 .item` |
| 1 | 0.0004 | `job.py:826 .item` |

The four `job.py` readbacks — the ones this plan targets — cost **3.8 us
per step in total**. §3.1 is already shipped (`generator.py:1102-1128`)
and free.

### 7.3 The step is device-bound with zero slack

Two direct experiments, interleaved, 6 windows each:

| arm | ms/step | tok/s | Δ |
|---|---|---|---|
| base | 17.344 | 57.66 | — |
| **nosync** (generator's terminal sync deleted outright) | **17.061** | **58.61** | **−0.283 ms, +1.63%** |
| hostload (+5 ms of pure-python spin at the step boundary, where the device is provably idle) | 22.35 | — | **+5.01 ms: not one microsecond absorbed** |

`hostload` is the decisive one: 5 ms of host-only time costs 5 ms of
step. There is no per-step device slack, so no host-side cut — batching
reads, moving the sync, or moving sampling into the graph — can buy back
host time. The host's real CPU work is ~11.7 ms against ~17.2 ms of
device work.

The 4.63 ms of fused-store status reads is the largest host-blocking
item in the step, but it is a *symptom* of device-boundedness, not an
independent cost: those reads block because the device is behind. Make
them free and the host simply waits at the terminal sync instead.
(That readback remains a real candidate for a *device-side* fix — it is
the "HARD, drives 0/2/1 control flow" item in
`wiki/reports/2026-10-02-tg64-host-bubbles.md` — but it is a different
mechanism from this task and out of its scope.)

### 7.4 Why §3.2 is worth ~0 and §3.3 is worth at most +1.63%

- **§3.2 (replace the full `synchronize` with a stream-event wait):
  worth ~0.** The wait primitive is not the cost. Measured on the box:
  `torch.cuda.synchronize(dev)` on an idle device is **3.9-4.1 us**, and
  `tiny.add_(1); torch.cuda.synchronize(dev)` — a full kernel+sync round
  trip — is **11.8 us**. The terminal sync blocks 0.667 ms because the
  device still owes ~0.65 ms of real work when the host arrives (device
  span from end-of-forward to the sampler tail is only 0.024 ms, but the
  forward's own tail is still in flight). An event wait on the same work
  blocks the same time. The sync is also not "for timing" and not
  cross-stream: the token is produced by kernels queued behind the
  forward on the same stream, so waiting for the token *is* waiting for
  the forward.
- **§3.3 (sample in graph): at most +1.63%, and only that.** The `nosync`
  arm is the plan's own ceiling — it removes the terminal wait outright,
  at the price of correctness (it feeds stale tokens). It recovers
  0.283 ms because the host stops waiting and the device absorbs the
  difference. A legal version of the same mechanism (token stays on
  device, input staging moves to device, no host stall) is bounded by the
  same number. The RNG problem is concrete: the seed is a host int drawn
  per step (`job.py:578`) and passed into `sampler.forward`
  (`sampler/custom.py:1185-1190`), so an in-graph sampler needs static
  Philox state and a device-side seed. Plus stop-string / EOS checks
  still need the value on the host, so the bookkeeping cannot simply
  disappear.

+1.63% is inside this box's own window-to-window spread (this run saw
22.3 ms windows against a 17.3 ms median, −22%), and it is only
reachable by the attack the plan ranks last and flags as high-risk.
The §2.2 rationale — "the prize is below the bar and the RNG-in-graph
risk is not worth it" — holds on the recoverable number.

### 7.5 Premise corrections

- **"~1.7 ms is loop overhead ABOVE the model — your prize pool ceiling"
  is stale.** Measured host time outside the model forward is 0.245 ms
  of staging/bookkeeping plus a 0.667 ms wait that is device latency:
  0.9 ms total, of which 0.283 ms is recoverable.
- **`generator.py:670` is the draft path**, not the main greedy sample
  (see §0). The main loop samples at `job.receive_logits`
  (`job.py:571-583`).
- **§3.1's "batch the `.item()` reads" already exists** upstream at
  `generator.py:1102-1128`, and the remaining reads are free.
- `DefaultSampler` lives at `exllamav3/generator/sampler/presets.py:3`,
  not `job.py:166-168`; `job.py:166-168` is the `sampler` ctor arg.
- The 4.63 ms fused-store status readback (`kvarn_triton.py:502`) is the
  real host-blocking item in a server decode step. It was previously
  estimated at "2 launches + 1 sync" per layer with no number attached;
  now it has one: **4.63 ms/step, 27% of the step's wall time**, spent
  blocked. Worth a task of its own if the store path is ever in scope.

### 7.6 Side finding, not chased

The server harness measures **57.66 tok/s at 64k** where the ledger's
microkld harness measures 47.8 for the same model/cache/protocol — the
server path is ~20% faster than the microkld path. Same model, same
env; the difference is the harness (generator `block_table` +
`pinned_staging` vs microkld's `past_len`), most likely a graph-bucket
key that stays stable on one path and misses on the other. Not chased
here (out of scope, and unverified), but it means the ledger number
understates the server by ~20% and any future server-path work should be
measured on the server harness.
