# Plan — closing the adaptive half of the expert tier

Status: **T1a and T2 are done; T3 and T5 are cancelled** (see below). Written 2026-11-07 after
`T50-why-the-transfer-failed.md`.

## Result that closed the plan (2026-11-07)

`T51-adaptive-placement-at-262k.md`: with the sweep on, coverage in the served regime climbs **22.3 % →
57.5 %** inside one pass (512 swaps), and decode does **not** follow — median single-stream 34.48 vs 36.37
for the static arm (−5.2 %), **37.40 vs 38.62 at 261 888 (−3.2 %)**. So the coverage mechanism works and
coverage is not the critical path at 262 144. **T3 (per-layer pool) and T5 (pool arm) are cancelled**: they
would raise a quantity that no longer moves the metric. What the result points at is GPU-side per-row work,
which has no instrument — that becomes T7.


## What this plan is for

T50 established three things that decide the design:

- The fork's VRAM tier is **25.6 % of experts and covers 31.76 %** of routing; Strata's is **33.8 % and covers
  82.77 %**. Eight more points of residency buy fifty-one points of coverage, so residency is not the lever and
  no VRAM purchase can close the gap.
- The **static** halves are at parity (Strata's shipped profile: 28.4 % at 8,000 slots; ours: 31.76 % at 6,288,
  from a saved profile — the tier is *not* identity-filled). There is nothing to win by improving the ranking.
- The **adaptive** half is absent in production (`EXL3_MOE_CPU_PROMOTE=0` → `spares = 0`) and, where it exists,
  is shaped so it cannot pay: spares are carved out of the resident slice (`block_sparse_mlp_cpu.py:741`),
  6 swaps/step at decay 0.98 (`:161-163`) against Strata's 96 swaps/4 rounds at 0.7, a swap that costs
  ~386 µs against 41 µs of avoided host work, inline (≈9 % of the token), and a per-layer uniform prefix
  where Strata runs one global pool (its own per-layer variant costs it −15.5 %).

**Goal:** raise coverage from ~32 % to ≥55 % *at 262 144 cells* and convert that into decode tok/s, at zero
additional VRAM, without touching the mandatory context.

**Acceptance:** a same-session ABBA arm at `-cs 262144` in which the adaptive arm beats the reference by
≥ 15 % single-stream decode at 261 888 with every primary category ≥ 1.00, coverage ≥ 50 %, and the box
restored afterwards.

## Design sketch (to be firmed by T2)

1. **Per-layer pool, fixed total.** Replace the scalar `split_k` with a per-layer vector whose sum is
   unchanged (48 × 131 = 6,288 slots). Seam: `split_k = int(getattr(ip, "moe_cpu_split", 0))` at `:971`
   resolves per module already, and everything downstream uses `first = num_experts - split_k` per layer — so
   this is an **input** change, not a loader rewrite. Source of the vector: the saved expert profile's
   per-layer routing mass, with a floor and cap so no layer starves.
2. **Exchange, not admission.** A promotion should copy the incoming expert into the *coldest resident's* slot
   in place (what `_split_sweep_layer` already does) instead of admitting into a spare carved out of the tier.
   That removes the negative-sum trade at `:741` without needing VRAM: the slot count never changes.
3. **Cheaper swap (optional, needs measurement).** The promoted expert is already in the pinned host arena —
   the worker holds every CPU-resident expert — yet `_split_swap_experts` re-reads it from the checkpoint.
   Copying from the arena would target Strata's ~47 µs/swap rather than our ~386 µs. Gated on T1b.

## Tasks

| # | task | seat | needs box | deliverable | acceptance |
|---|---|---|---|---|---|
| **T1a** | Coverage **and sweep activity** in the served regime (`EXL3_MOE_TIER_STATS=1` + `EXL3_MOE_CPU_SWAP_DEBUG=1`, `-cs 262144`, Q4, DFlash2, pin on) | Flash | **yes — needs consent** | one pass + coverage and swaps-per-sweep | pooled coverage with n and regime; the swap count per sweep; gate pasted. If swaps/pass ≈ 0 the sweep is inert and T3 changes shape |
| **T1b** | Swap cost measured, not derived: time `_split_promote_step` per swap with the instrument on, at the served config | Flash | yes (same session as T1a if possible) | µs/swap, histogram | replaces the derived 386 µs with a measured number |
| **T2** | Design note: per-layer vector format (env vs config), allocation rule, exchange protocol, failure modes, exact seams | DeepSeek V4.1 Flash | no | `wiki/plans/2026-11-08-pool-shape-design.md` | every seam cited `file:line`; arithmetic proving Σ slots constant; a stated rollback |
| **T3** | Implement per-layer split + cadence/floor retune behind env vars, default off. **The exchange protocol already ships** (`_split_swap_experts`), so this is three deltas, not a new mechanism: the vector, a floor/cadence that spends the budget, and the optional arena-copy swap | DeepSeek / GLM Flash | no | code + tests | existing `tests/moe_cpu` green; new test proves (i) Σ slots constant, (ii) a promotion never reduces static residents, (iii) `reset_placement` still restores |
| **T4** | Review T2/T3 | Muse Spark 1.3 | no | review notes | findings triaged; no unresolved must-fix |
| **T5** | Arm: ABBA at `-cs 262144`, reference vs adaptive, `-dr 3`, coverage on both sides | Flash | yes | CSV rows + report `T51-pool-shape.md` | pre-registered prediction met or the mechanism is withdrawn; gate pasted |
| **T6** | Deploy or withdraw; restore box either way | me | yes | state doc + deployed config | port 5000, production config, `/v1/models` 200 |
| **T7** | Per-stage profile of one decode round: dense, mixer, router, GPU experts, **CPU-expert wait**, combine, drafter | DeepSeek Flash (design) | yes (one pass) | design + one profile | phases sum to the measured round within 10 %; names the largest phase at 131 k and 262 k |

T1a and T2 are independent and can run concurrently; T3 depends on T2; T5 depends on T3 and on T1a's baseline.

## Pre-registered predictions

Written before any arm, so a miss is information rather than a reinterpretation.

- **T1a:** served-regime coverage reads **26–32 %** — at or below 31.76 %, because a 5-row verify window
  widens the routed union while committed tokens only grow ~3.05×. Above 35 % ⇒ the instrument or the
  placement differs in the served config, and the T50 ledger needs re-deriving before T5.
- **T5:** coverage **≥ 50 %** (miss if < 45 %); single-stream decode at 261 888 **≥ 45 tok/s** against the
  same-session reference (miss if < 42, i.e. < +8 %); one primary category below 1.00 ⇒ withdraw even if the
  weighted primary passes.

## Seats and quota

The catalogue has **323 models across 12 provider entries**, not the 25 that a single 100-row page showed. The
table below is the corrected seat plan.

| role | model | price ($/M in–out) | why |
|---|---|---|---|
| hard tasks, design drafts, design review | `opencode/opencode/crouton-alpha-free` | **0 / 0** | operator-selected seat (2026-11-08); this session's own model |
| alternate implementer | `opencode/opencode/deepseek-v4.1-flash-free` | **0 / 0** | free tier of the same family; also GLM (`opencode/opencode/glm-5.3-flash-free`) when DeepSeek is rate-limited |
| cross-review of a crouton-alpha implementation | `opencode/opencode/deepseek-v4.1-flash-free` | **0 / 0** | a different model line reviews crouton-alpha's work |
| alternate implementer | `opencode/opencode/mimo-v2.6-flash-free` | **0 / 0** | transcription-only work |
| local model (the served weights) | `tabbyapi/tabbyapi/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8` | **0** (own box) | registered and reachable; **runs on the GPU that serves traffic**, so small jobs only, never during a measurement pass |
| cross-model review | a *different* free seat from the author (e.g. GLM reviewing DeepSeek) | **0 / 0** | disagreement between two free seats is the signal |
| tie-break review | `opencode-go/muse-spark-1.3-contributor` | 0.10 / 0.20 | one call, only when the free reviews disagree |
| go/no-go on the whole design | `gemini-4-argon` — **at most once**, only on request | 2 / 10 | excluded under the current quota |

Two corrections to what an earlier reading of the catalogue produced, both recorded so they are not repeated:
the local `Qwen3.8-Flash-Next` **is** registered as an opencode provider (`tabbyapi/…`), and free tiers of
DeepSeek, GLM and MiMo exist. The earlier conclusion came from reading one 100-row page of a 323-row
catalogue and ignoring the response's `total` and `next`; model discovery must page to `next == null`.

## Open questions (deliberately not settled here)

1. Per-layer vector on the command line/env or in `config.yml`? Env is cheaper to arm and revert; config is
   what ships. Leaning: env for T5, config key only if it ships.
2. Allocation rule: proportional to per-layer profile mass, or proportional to *observed miss mass* (needs
   T1a's data)? The second is the better-targeted one and costs nothing extra if T1a's instrument can print
   per layer (`EXL3_MOE_TIER_STATS=2`).
3. Does the exchange protocol need the hysteresis/floor logic from `_split_sweep_layer`, or a simpler
   "promote only if the incoming expert's count exceeds the victim's by 2×"?
4. Whether the drafter has to be re-tuned after placement changes — Strata recorded that moving tier contents
   perturbs its drafter (its forecast arm was a wash for exactly that reason).

## Risks

- **Coverage may not convert.** `t29-C1` reached 72 % coverage and bought no decode — but at 16 spares (115
  static residents), 6/step, inline, pre-drafter. T5 must isolate the shape, and T1b exists to explain a null.
- **Per-layer allocation can regress a layer** that the uniform split served well; the floor/cap is the guard.
- **A swap that is not cheap is not worth having**; if T1b measures ≥ 300 µs/swap, T3 must land the arena-copy
  path first or the arm will reproduce the old null.
- **Quota**: every task above is scoped to one subagent and one artifact, so a rate limit costs a retry, not a
  campaign.

## Not in scope

- Cache shrinking in any form (262 k mandatory).
- Growing the tier (allocator refuses; measured worth +4.4 % for +21 experts).
- Porting Strata's profile format, its MTP drafter, or its PCIe miss share (all measured or closed in T31/T50).

## Operator decision, 2026-11-07: no lower bitrate

The per-layer 2 bpw tail is **not to be pursued as a performance lever**. Strata reaches its speed with a
*larger* blob per expert (IQ3_S, ~2.046 MB) than ours (3.18 bpw, ~1.95 MB), so lower bpw is not how Strata wins
and copying it would change the model's quality basis while diverging from the reference. This supersedes the
"postponed on quality" status: the lever is **dropped**, not parked.

Consequence for the plan: the CPU tail (12 ms of the 52 ms round) can only be **hidden**, not cheapened —
and the arithmetic below says it was never the main gap anyway.
