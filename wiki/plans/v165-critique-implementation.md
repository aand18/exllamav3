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
| C3 | The in-run `Baseline` is confounded by (a) dynamic expert swap sweeps between arms and (b) the 51.8 GB ngram table streaming from disk | **Confirmed** as mechanism (`run_pending_swap_sweeps` every 128 steps; PLE streams when `ngram_ram` off); explains the 6.99 t/s baseline outlier | Protocol: never divide by a cold/streaming baseline. Set `EXL3_MOE_CPU_SWAP=0` for knob arms and warm the ngram path before timing; drafting verdicts use `ARM` lines | ✅ (protocol; the harness half was false until `7c043454` + `8c69b5ec` — see D2/D4) |
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
| I5 | Single machine-readable state; no copy-pasted CLI strings | `bench/windows/state.json` is the one source (`run_pass.ps1` reads model/split/threads/cache/env/draft args from it; `-Extra`/`-Env` carry only the delta). Plans express steps as deltas vs that file. `spec_decode` results land in `results/spec_decode.csv`, and `ratios.py` reads it | ✅ (draft args and the drafting cache only since `7c043454`/`89f0b787` — see D2) |
| I6 | Automated pre/post-task gates so a report cannot contradict its own logs | `bench/verify_task.py`: pre-run (wheel sha, profile warm, `state.json` consistency, `size_check` fit) and post-run (every cited `run_id` is `ok`, `breach=0`, profile-warm header present, `ratios.py` table regenerated) | ✅ |

## C. Second review (code + skills)

Second review of this work: session `ses_e762fc6aaffeqwXTkhBgyOdswh`, message
`msg_18a21ae37001YArivM6a5bJ4zw` ("Qwen3.8-Flash-Next retuning plan critique"), read against the
implementer's report (`msg_18a04de20001iGNMGh2WCdCcDY`) and its skill proposal
(`msg_18a123bb3001Zw9IGKxG3M8W98`). Part 1 found six defects the first pass missed and all six are
fixed on this branch; Part 2 ruled on four skill proposals and those rulings are binding (one skill
extended, nothing else created). Every commit named below is on `165-win-perf` between `79926691` and
`1c5b9670`. The sections that follow C ("Conflicts resolved", "Verification performed", "Not
exercised yet") describe the **first** pass; C records what the second review changed on top. The
documentation that carries C is two commits of its own: `7c8754e2` (`bench/windows/README.md`, the
harness as it now is) and `dfa36468` (this file). The implementers' evidence for every row — dry-run
transcripts, mutation matrices, test counts — is archived beside the run logs under
`wiki/plans/v165-reports/artifacts/second-review/`.

| # | Defect (the review's claim) | Evidence | Fix | Commit | Status |
|---|---|---|---|---|---|
| D1 | `deploy.sh` ships `state.json` only when the destination has none, so a regime change in the repo is not deployed while `verify_task.py --pre` validates the new one | `deploy.sh:25-27` (pre-fix) installs under `if [ ! -f ... ]` | `state.json` installed on every run; `state.json: updated` / `unchanged` printed from a sha256 before/after; the file joins the closing `sha256sum` | `5134738b` | ✅ (four deploys to a scratch `DEST`: updated, unchanged, updated on a regime edit, updated on the revert) |
| D2 | `run_pass.ps1` never read `state.json`'s `.draft` or `.knob_arms_disable_swap`: a drafting arm ran the engine default 4 draft tokens while the state said 2, and the drafting cache arrived as a **second** `-cs` | `perf-val-spec-20261029-233352.log:160-163` logs `-cs 131072 … -mtp -cs 40960` and `ARM … acc=1.74/4.00`; its env line has no `EXL3_MOE_CPU_SWAP` | `$ST.draft` → `-mtp/-ndt/-sdh/-hdt/-dds` (`spec_decode` only), `-CacheSize` (`draft.spec_decode_cache_size` for the drafting runner), one `-cs`, `EXL3_MOE_CPU_SWAP=0`, `-DryRun`; `-Extra` may not restate a regime flag in any spelling | `7c043454`, `89f0b787`, `8c69b5ec` | ✅ (dry run: `-cs 40960` once, `-mtp -ndt 2`, `EXL3_MOE_CPU_SWAP=0`; a real drafting pass is the T6 resume) |
| D3 | `warm_corpus()` read the prompts without the `tool_calls[i].function.arguments` decoding `measure()` does, so the warm-up templated a string where the measurement templated a dict | `eval/spec_decode.py:184-191` vs `:52-58` (pre-fix) | one `read_prompt()` for both callers, plus a test that parses every file in `eval/prompts/` and pins both call sites | `c6daa77d` | ✅ |
| D4 | Dynamic expert placement makes an arm-to-arm ratio meaningless: with `EXL3_MOE_CPU_SWAP` on, the arms do not generate the same tokens | `block_sparse_mlp_cpu.py:830` decides at load and `:957-960` says a placement change "visibly derails a generation in progress" (`:955` is the `EXL3_MOE_CPU_SWAP_INTERVAL` guard); `perf-val-spec-…log:61` vs `:103` differ from the same prompt | `swap_refusal(arms, dynamic)` (pure, tested) + `dynamic_placement()` read after `model_init.init`; a multi-arm run with placement on exits 2 unless `-swap_ok`; the harness sets `EXL3_MOE_CPU_SWAP=0` for the drafting runner so the gate should never have to fire; a `spec_decode` run with no `ARM` line and no table row is `fail`, so a refusal cannot land as `ok` | `c6daa77d`, `db1029e3`, `25d7a6e0` | 🟡 (the pure gate and its wiring — `swap_refusal`, `count_arms`, `dynamic_placement`, the `main()` block — and `verify_task.arm_gate` are covered by `bench/test_gates.py`; the `run_pass.ps1` verdict branch ran nowhere, verified by `Parser::ParseFile` plus a code read; no multi-arm run has loaded a model since) |
| D5 | `alloc peak` printed before `measure_generate`, so a `-spf` probe reported allocation before generation and `passes.csv` recorded `0,0` for spec_decode | `eval/perf.py:209` (pre-fix); `passes.csv:45` | the print moved to the last statement of `main()` in `eval/perf.py` and added to the end of `eval/spec_decode.py`, same wording (`alloc peak allocated=<n> MiB reserved=<n> MiB`) | `c6daa77d` | 🟡 (the next real pass fills `peak_alloc_mb`/`peak_reserved_mb`; nothing offline can) |
| D6 | `verify_task.py`'s gates were no-ops: the warm test matched the string anywhere in the log, the percentage gate printed PASS for any `%`, `--pre` called `size_check.py` without `--draft`, and nothing compared the deployed `state.json` with the repo's | `perf-mtp-ref-318-20261029-222244.log` has 49 `CPU split experts` lines, 48 warm, line 2 (`mtp.layers.0.mlp`) cold, and the old substring test returned PASS; `verify_task.py:152-154` (pre-fix) | per-line warm gate naming every cold line, ratio gate that reads the report's own table min/max, `--draft mtp` from the state, `--deployed` split-brain gate, arm-row gate; 34 stdlib tests | `ffec2e50`, `1c5b9670` | ✅ (`--pre` passes on the repo state; `--post` on `T06-mtp.md` fails on that cold line, a `±` band and two misquoted ratio bands, and the report stays as written) |

Two rows in A and B claimed more than the code did at the time. They are true now, and only because
of the commits above:

- **C3** claimed "protocol **+ harness**" for `EXL3_MOE_CPU_SWAP=0` while `run_pass.ps1` never set it.
  True since `7c043454` (written into the regime env, before `-Env`/`-Unset`) and `8c69b5ec` (scoped:
  automatic for `-Runner spec_decode`, opt-in with `-StaticSwap`, because the 44 archived ladder rows
  were measured with placement on).
- **I5** claimed `run_pass.ps1` read "draft args" from `state.json` while it read neither `.draft`
  nor the drafting cache. True since `7c043454` (the `.draft` block) and `89f0b787` (draft args are
  `spec_decode`-only; `-CacheSize` carries `draft.spec_decode_cache_size`).

### The four skill proposals

| # | Proposal | Verdict | Why |
|---|---|---|---|
| S1 | new skill `benchmark-harness-integrity` | **not created** | same trigger as `measurement-protocol`, and four of its five bullets were code defects, not agent judgment: they are D1–D6's territory, and a skill cannot make a harness enforce what its code did not do |
| S2 | new skill `fit-before-measure` | **folded into `measurement-protocol`** as one section, "The fit comes before the pass" | same trigger branch, and a second model-invoked skill for one workflow adds permanent context load for a rule that fits in a section |
| S3 | extend `measurement-protocol` instead | **extended** — the only skill change made | that skill already owns this workflow, and the review's instruction was to extend it and create nothing |
| S4 | new skill `integrating-external-critique` | **rejected** | duplicates the existing `receiving-code-review` skill, and its pitch contained a false claim about what the critique had said |

### S3 — what the skill now says, and the evidence it was needed

`~/.config/opencode/skills/measurement-protocol/SKILL.md`, 71 → 99 lines; the frontmatter
`description` is unchanged (the skill already triggers on this workflow). No commit: the skills
directory is not a git repository, so the file is the record. Three sections:

- **The fit comes before the pass** — `bench/size_check.py` agrees on the configuration before a GPU
  pass launches; both curves (VRAM and host arena against the same knob) are plotted before an
  operating point is named; each margin (watchdog floor, `EXL3_HOST_MEM_RESERVE_MB`, the autosplit's
  reserve) is named and charged once; a "does not fit" verdict names its phase (load, warmup, a
  prefill band, decode).
- **The stateful-cache table** in "A recorded number is not a control" — the routed-expert profile
  (`expert_stats`), the sparse-head token profile (`token_stats`, keyed on the tokenizer hash and the
  vocab size, so one file spans every quant **and** every `-sdh` size), the 51.8 GB PLE n-gram table
  streamed from disk, and dynamic placement.
- **The gates, before and after** — `verify_task.py --pre` before the first pass of a task,
  `--post --report <file>` before a number is quoted, and exit 1 means the report contradicts its
  logs, not that the gate needs weakening.

RED/GREEN, five fresh-context samples per arm (one RED sample discarded as contaminated by the
prompt, so the RED counts are out of four): the old skill had **0/4** naming `size_check.py` and
**0/4** naming `verify_task.py --pre/--post`, while already freezing the profile and refusing
recorded baselines 4/4; the new one ran the pre-gate before launching **5/5** and named `--post`
before quoting **5/5**, with no regression on the state or baseline rules. The sections exist
because those two behaviours were absent, not because the text was thin.

### Documentation these commits changed, and the spellings that now abort

- `bench/windows/README.md` now describes this harness: `-DryRun`, `-CacheSize`, `-StaticSwap`,
  `-Model` and `-Runner` documented, the regime moved from "fixed in `run_pass.ps1`" to
  `state.json` (with the drafting rule and the frozen-profile restore), `-Extra "-cq Q8"` replaced
  as an example, the `_agent/v164` path corrected, and the outputs and `status` rules brought up to
  what the parser and the verdict produce. Every command in it was checked against the `param`
  block, and the examples were run with `-DryRun`.
- `wiki/plans/v165-reports/T06-mtp.md:10` records `-Extra "-rb_ab -mtp"`, and
  `wiki/plans/2026-10-29-v165-retune.md:206,511` records `-Extra "-mtp -ndt 2"` and
  `-Extra "-rb_ab -mtp"`; both abort in the current harness. They are **not** rewritten — they are
  the record of the runs that produced the numbers. `state.json` carries the draft args
  (`-mtp -ndt 2`), so the replacement spelling for a readback A/B is `-Extra "-rb_ab"`: `-Extra`
  carries only the delta.

### Still unrun after this plan (the honest 🟡 boundary)

- **T6 resume** — a multi-arm in-process drafting run: the first real `-Runner spec_decode` pass
  with `-mtp -ndt 2` at `-cs 40960` (D2), the first run the swap gate can refuse (D4), and the first
  `peak_alloc_mb`/`peak_reserved_mb` values a drafting pass writes (D5).
- **T7 sidecars** — after T8, because `-mtp -or` needs the `1385306e` assert relaxation that only the
  win-perf wheel carries (C12).
- **T10 ladder** — rewritten and unrun; it is read against the frozen profile, so it needs C2's warm
  re-measurement first.
- **T13** — 262144 together with the drafter (`DRAFT_WINNER`), and the hugepage measurements (the
  12–19 % tg cost of `EXL3_MOE_ARENA_HUGEPAGE=0`), are measured rather than assumed.

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
