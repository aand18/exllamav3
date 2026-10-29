# v165 critique + improvements — implementation tracker

Source: two messages from session `ses_e762fc6aaffeqwXTkhBgyOdswh` ("retuning plan critique",
`msg_189d5966f001n4kRJKiCMY2sbs` and `msg_189d970b5001OTvAYPuC7sgOQM`).
Operator instruction: implement **everything** in both lists; where they conflict, the
**Improvements prevail**; nothing may be silently dropped.

Every item below was checked against the code or the archived logs before acting. "Verdict" is
what the evidence says about the critique's claim; "Action" is what this repo now does.

Legend: ✅ done · 🟡 implemented, GPU validation pending · ⚠️ confirmed but deliberately deferred
(recorded why)

---

## A. Critique items

| # | Item | Verdict | Action | Status |
|---|---|---|---|---|
| C1 | `mtp-ref-318` was a **cold** run (no saved expert profile); later arms loaded `placement from saved profile`, so the T6 ratios divide warm arms by a cold reference | **Confirmed** — `perf-mtp-ref-318-…log:2` has no `placement from saved profile`; `mtp-ndt2-318:2` does | T6 report demotes `mtp-ref-318` to "cold reference"; the reference must be a **warm** re-run (`mtp-ref2-318`) and the verdict read from the `ARM` lines (tok/s + acceptance), not from `MTP/Baseline` | ✅ (report + protocol) |
| C2 | The 3.18 split ladder was measured on a cold first load and warm subsequent loads; "8 experts = +15 %" was never ABBA'd | **Confirmed** — `smoke318` was the first 3.18 load; `split376*` show `placement from saved profile` | Ladder result is recorded as **provisional**; warm re-run + ABBA required before SPLIT is frozen; the split gate now runs on a frozen profile (`state_snapshot.sh`) | ✅ (documents) / 🟡 (re-measure) |
| C3 | The in-run `Baseline` is confounded by (a) dynamic expert swap sweeps between arms and (b) the 51.8 GB ngram table streaming from disk | **Confirmed** as mechanism (`run_pending_swap_sweeps` every 128 steps; PLE streams when `ngram_ram` off); explains the 6.99 t/s baseline outlier | Protocol: never divide by a cold/streaming baseline. Set `EXL3_MOE_CPU_SWAP=0` for knob arms and warm the ngram path before timing; drafting verdicts use `ARM` lines | ✅ (protocol + harness) |
| C4 | The `-sdh` "profile build, discard" plan never measures the discarded size, and the token profile is shared across checkpoints and sizes | **Confirmed** — `token_stats_path` keys on tokenizer+vocab, not on checkpoint or `-sdh`; profile only observed while an `sdh>0` generator runs | Plan text rewritten: profile warm-up run is **not** a measurement of any size; each `-sdh` size gets its own measured run after the profile is warm | ✅ |
| C5 | T4 spread is misquoted as `pp ±5 % (0.95–0.99)` when the 4096 band is `0.92` (A2 slower everywhere) | **Confirmed** — `T04b` ratio table: pp 0.92 at 4096, every band ≤0.99; tg 0.91–1.01 | T04 and STATE rewritten: the pair is a **one-sided drift** (`A2 = 0.92–0.99 × A1`), so the threshold is an interval, not a symmetric ± | ✅ |
| C6 | "Drafting leaks ~700 MB per arm" is **self-inflicted**: `spec_decode.py` builds 5 `Generator`s without clearing `recurrent_cache`/`host_pool`; tabbyAPI creates **one** generator at load | **Confirmed** on both counts — Generator is inside the arm loop; tabbyAPI has `create_generator()` + `close()` + `gc.collect()` on its error path | Report corrected (no engine leak, no tabbyAPI "per job" caveat); `spec_decode.py` gains `reset_arm()` and reuses one generator for the readback toggle | ✅ |
| C7 | "split 368 does not load at all (15 MB)" is false: it loaded, warmed up and ran to `Prefill:` | **Confirmed** — `perf-split368-131k-…log` reaches `Prefill:`; `passes.csv` has `split368` **ok** at 4096 with 2045 MB free | STATE/T04 corrected: 368 **loads**, and fails during long prefill on an allocator scratch request; "does not load" withdrawn | ✅ |
| C8 | "Large pages (no privilege on this box)" is false — the box does have `SeLockMemoryPrivilege` | **Confirmed** — `moe_cpu_host.py:_win32_enable_lock_memory_privilege`, and logs show GBs committed on large pages | Removed from Advise-only; `EXL3_MOE_ARENA_HUGEPAGE=0` is documented as a **benchmark control only**, and T13 must record the 12–19 % tg cost of shipping it | ✅ |
| C9 | "The load transient dips ~2 GB deeper for `max_length ≥ 131072`" is false — the loader never sees `max_length`; the difference was background host-RAM drift | **Confirmed** — same split, 4096 vs 131072 watch traces dip 8063 vs 8382 MB | Plan claim withdrawn and replaced by "host RAM drifts between runs; floors + frozen state, not `max_length`, decide" | ✅ |
| C10 | T13 fit math is wrong: the MTP head is **1 layer** with **0 GDN layers**, so its 262k Q2 draft cache is ~156 MiB, not 1.1–2.1 GB; `perf.py -sd` allocates no second cache at all | **Confirmed** — `mtp.layers.0.mlp` in every log is the only MTP layer; per full-attn layer = 1.875 GiB / 12 = 160 MiB at 262144 Q2; `perf.py -sd` just loops seqlens on one cache | T04/T06/STATE math replaced; T13 must **measure** 262144 + `DRAFT_WINNER` at split 376–384 instead of assuming it cannot fit | ✅ |
| C11 | tabbyAPI: `draft_mode: mtp` reuses the main config (so the head inherits `moe_cpu_split`); `draft_mode: model` builds a fresh config and drops `draft_moe_cpu_offload`/threads | **Confirmed** — `model.py` `if draft_mode == "mtp": self.draft_config = self.config` | Recorded in STATE as a deployment fact for T13 (the sidecar/DFlash2 path needs the offload knobs forwarded, the MTP path does not) | ✅ |
| C12 | **T7 trap:** `-mtp -or` needs the `1385306e` assert relaxation, which lives only in the win-perf wheel that T8 installs *after* T7 | **Confirmed** — installed stock wheel has `assert not draft_model_dir, "Tensor overrides not supported..."` at `model_init.py:263` | Plan reordered: **T8 before T7** (wheel A/B first, then sidecars), with the alternative (copy the one-line Python patch into the installed stock package) written down | ✅ |
| C13 | Harness bugs: duplicate `-cs`; no structured spec_decode metrics; stale `$Tag` in the BREACH echo; `v164` defaults in four places | **Confirmed** — `run_pass.ps1:131` hardcodes `-cs 262144`; `:372` echoes `$Tag`; `:56`, `run_build.ps1:21`, `watch.ps1:40`, `deploy.sh:19` default to `_agent\v164` | All four fixed: cache size comes from `state.json`; spec_decode runs write `results/spec_decode.csv`; BREACH echo uses `$runId`; every default points at `v165` (and a wrong default now aborts instead of writing to a retired dir) | ✅ |
| C14 | T9's thread/stream knobs are CPU↔GPU balance knobs, so screening them on the 1.66 model can invert the answer; `EXL3_SPLIT_PROF=1` was never used | **Confirmed** (balance differs: 136/376 at 3.18 vs 152/360 at 1.66) | T9 rewritten to run on **3.18**, gated by a 60 s `EXL3_SPLIT_PROF=1` probe (wait/issue counters) before any full ladder | ✅ |
| C15 | T10's text still instructs the 3.68 / 262144 ladder 400→448→480 | **Confirmed** | T10 rewritten for 3.18 (`368/372/376/380` + `-mcpa` + promote-on-miss), with the measured boundary (368 loads but dies in prefill) | ✅ |

## B. Improvement items

| # | Improvement | Action | Status |
|---|---|---|---|
| I1 | Pre-flight static geometry check, no GPU: exact VRAM + arena bytes per `-mcs`, drafter footprint, minimum `-cs` | `bench/size_check.py` (header-only, prints JSON + table). No GPU pass may launch until it agrees with the budgets | ✅ |
| I2 | Freeze hidden state across arms: expert/token profile snapshots, and a clean multi-arm process | `bench/state_snapshot.sh` (snapshot/restore via `XDG_CACHE_HOME`, which pins both `expert_stats` and `token_stats`); `eval/spec_decode.py` gains `reset_arm()` and one shared generator; protocol: restore the snapshot before every arm, `EXL3_MOE_CPU_SWAP=0` while comparing knobs, warm the prompt corpus first | ✅ |
| I3 | Watchdog/allocator interaction: flush cached blocks between bands, record allocator watermarks, watch the whole process tree | `perf.py` calls `torch.cuda.empty_cache()` between bands and prints `alloc peak allocated/reserved`; `run_pass.ps1` parses those into `passes.csv`; `watch.ps1` sums the working set of the target PID **and its descendants** (the 43 GB arena lives in the worker) | ✅ |
| I4 | Mechanism counters first, tok/s second | Protocol + tooling: `EXL3_SPLIT_PROF=1` 60 s probe gates thread/stream knobs (T9/T10); drafting separates deterministic acceptance (1 pass) from step latency (2–3 fixed-length workloads, one process) | ✅ |
| I5 | Single machine-readable state; no copy-pasted CLI strings | `bench/windows/state.json` is the one source (`run_pass.ps1` reads model/split/threads/cache/env/draft args from it; `-Extra`/`-Env` carry only the delta). Plans express steps as deltas vs that file. `spec_decode` results land in `results/spec_decode.csv`, and `ratios.py` reads it | ✅ |
| I6 | Automated pre/post-task gates so a report cannot contradict its own logs | `bench/verify_task.py`: pre-run (wheel sha, profile warm, `state.json` consistency, `size_check` fit) and post-run (every cited `run_id` is `ok`, `breach=0`, profile-warm header present, `ratios.py` table regenerated) | ✅ |

## Conflicts resolved (Improvements prevail)

1. The critique wants `mtp-ref-318` re-run warm as an ABBA bracket; I2 wants a frozen profile
   restored before every arm. **Resolution:** freeze once, restore per arm, and the ABBA bracket
   becomes unnecessary *and* cheaper — the reference and the arm start from identical state.
2. The critique suggests `EXL3_MOE_CPU_SPLIT_STATS=<frozen.json>`; no such env var exists at
   v1.6.5 (`EXPERT_STATS_DIR` is a module global). **Resolution (I2):** use `XDG_CACHE_HOME` to
   point both stat caches at a frozen directory — env-only, no engine patch.
3. The critique's T6 fix re-runs arms for acceptance; I4 says acceptance is deterministic and
   needs one pass. **Resolution:** acceptance is read from `ARM` lines in a single warm run per
   arm; ABBA is only for latency claims.

## Verification performed

All of it is reproducible from the repo; nothing below is an assertion.

| check | result |
|---|---|
| `size_check.py` vs the measured cache | 2.093 GiB at 262144/2 slots — matches the probe exactly |
| `size_check.py` vs a measured pass | predicted 2704 MiB free VRAM; `mtp-ref-318` measured **2761 MB** (Δ 57 MiB) |
| `size_check.py` vs the engine's own refusal | `--min-cs 34560 1024` → 35584 tokens = **139 pages**, exactly the engine's `Job requires 139 pages` |
| `size_check.py` on the T13 question | 262144 + MTP at split 376 → 20852 MiB used, **1664 MiB free after reserve**, FITS |
| `verify_task.py --pre` | state/model/wheel/profile/fit gates run; `--expect-or` fails on the stock wheel (the T7 trap) |
| `verify_task.py --post` | reads run ids out of a report and checks `status=ok`, `breach=0`, warm-placement header |
| `ratios.py` on `spec_decode.csv` | reconstructed 112 rows from logs; ndt2 beats the default arm **1.30× median**, ndt6 0.97× |
| harness end-to-end (`val-state`, 4096 bands) | state.json defaults used with no `-Model`/`-Split`; `ALLOC peak_allocated_mb=19666 peak_reserved_mb=20592`; **`peak_proc_ws_mb=38242`** (was 3.7 GB before the tree fix); env column shows `FROZEN_STATE=master->active` |
| frozen state | `engine-state/active/exllamav3/expert_stats` repopulated from master before the run; master holds the two model profiles (49-layer MTP-bearing and 48-layer) |
| deploy defaults | `deploy.sh`, `run_pass.ps1`, `run_build.ps1`, `watch.ps1` all default to `_agent\v165`; no `v164` left except a historical plan reference |
| PowerShell syntax | `run_pass.ps1`, `watch.ps1`, `run_build.ps1` parse clean after every edit |

| drafting path end to end (`val-spec`, one workload, 128 tokens) | `ARM baseline tok_s=45.14 acc=0.00/0.00` and `ARM draft tok_s=52.50 acc=1.74/4.00` in `results/arms.csv`; `results/spec_decode.csv` holds `Coding, Baseline 45.14` and `Coding, MTP (greedy) 52.50, 1.16x, 1.74/4.00`; `ratios.py` reads the table back |

## Not exercised yet (deliberately)

- A *multi-arm* in-process drafting run (the thing `reset_arm()` and the shared Generator were
  built for) belongs to the resumed T6 series; the single-arm path is proven above.
- T9's `EXL3_SPLIT_PROF=1` gate is a protocol change (plan text), not code: the tool (`-Prof` on the
  harness, which sets `EXL3_SPLIT_PROF`/`EXL3_MOE_CPU_PROF`/`EXL3_MOE_SUBMIT_PROF`) already existed.
- T10's ladder is rewritten but unrun; it needs the frozen-profile re-measurement that C2 demands.
