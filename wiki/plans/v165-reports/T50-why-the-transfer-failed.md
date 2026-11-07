# T50 — Why the Strata transfer has not produced decode speed

**Question.** Strata decodes this model ~1.65× faster on this box (56.9 → 93.6 tok/s median, both drafted,
prompt ≥ 100 k; `T14-config-catalog.md:45`; Strata's served log of 2026-10-29, 17:23). We transferred six
mechanisms, shipped one (the pin, prefill only), and withdrew or refuted the rest. Why?

**Answer in one paragraph.** The fork implemented the *static* half of the coverage mechanism and is at parity
there; the entire remaining gap is the *adaptive* half, which the fork has code for but has never run in a
shape that can pay — and which the deployed config switches off. Strata's own numbers show the split: its
shipped static profile covers **28.4 %** of reads at 8,000 slots, its running tier covers **94.1 %** on the
same sweep (`/tmp/opencode/strata-vs-exllama-research.md:103-119`). Two thirds of its coverage — and 51 of its
82.77 points against our 31.76 — come from the tier *following the conversation*. That is the part we have
not got.

## 1. What we actually have, by coverage not by mechanism

| | experts resident | measured coverage | basis |
|---|---|---|---|
| fork, deployed config | 131 / 512 per layer = **25.6 %** | **31.76 %** | load line `[131..512) of 512, placement from saved profile`; `t29-A1-pin-off-20261103-220854`, instrument counts `(row, expert)` evaluations |
| Strata, live | 8,301 / 24,576 = **33.8 %** | **82.77 %** pooled, 82.20 % median (88 requests) | its served log of 2026-10-29, line 399 and the per-request hit-rate lines |

**Strata has 8 more points of residency than we do and 51 more points of coverage.** No quantity of VRAM can
produce that: coverage per resident expert is 2.6× better in Strata at a tier only a third larger. This single
comparison is the answer to "why more experts did not work" — residency was never the lever.

The one measured size lever confirms the scale: **+21 experts/layer = +4.1 % of experts → +4.4 % decode**
(`t49-C-128k-mcs360-20261106-194845` vs `t49-A-ref128k-20261106-193956`, 41.03 vs 39.31 at 130 816). Buying
every slot on the card would be worth single-digit percent, not 65.

## 2. The static half is implemented, and it is at parity

A first-pass reading of the source concluded the tier is filled by an identity map because
`EXL3_MOE_CPU_SPLIT_STATS` is unset. **That reading is wrong for the deployed config**, and the box says so:

```
-- CPU split experts (worker, dynamic): model.language_model.layers.0.mlp [131..512) of 512, placement from saved profile
```

The profile is real and on disk: `C:\Users\yoho\.cache\exllamav3\expert_stats\50ac90eb5104cb75.json`
(key = `sha1("D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8|512")[:16]`), written **2026-11-04 20:19**,
49 layer entries, 82,274 total decayed hits. `_split_perm` permutes the router hot-to-cold at load
(`block_sparse_mlp_cpu.py:1179`, `:1338-1389`, applied `:1456`).

So the ranking *is* ported. It is also, per slot, better than Strata's: 25.6 % residency → 31.76 % coverage
(1.24 coverage per resident point) against Strata's static 32.6 % → 28.4 % (0.87). **There is nothing to win
by porting the profile harder.** Caveat worth recording: the saved counts are decayed (`:78-100` writes
`_split_hist`, decay 0.98/step, half-life ≈ 34 steps), so our profile is a snapshot of recent traffic rather
than a long-horizon frequency distribution — and it still beats Strata's shipped static profile.

## 3. The adaptive half: present, and shaped so it cannot pay

Strata: one global arena, `next_free_` as the only admission counter with `per_layer_` off
(`include/strata/core/expert_cache.hpp:213-222`); adaptation every 4 rounds, **96 swaps**, decay **0.7**
(`src/program/generate.cpp:474-497`), the candidate and victim lists collected per layer and truncated as
**one list across the whole model** (`:7977-7998`), on its own thread with async copies
(`:8015-8017`, `:8132-8133`).

Fork, in five respects:

| # | fork | Strata | consequence |
|---|---|---|---|
| 1 | **spares are carved out of the resident slice**: `gpu_first = first - spares` (`:741`) | pool total fixed; a swap exchanges two residents | 16 spares leave **115** static residents instead of 131 — the churn pool is bought by permanently evicting hot experts |
| 2 | **6 promotions/step**, ratio 1.5, decay **0.98** (`:161-163`) | 96 swaps / 4 rounds, decay 0.7 | 16× smaller batch; with decay 0.98 a promoted expert's credit halves in ~34 steps |
| 3 | swap = **checkpoint read + in-place copy** (`_split_swap_experts`) | RAM↔VRAM exchange, no file reads | the code's own comment prices a transfer at **73 µs** against **41 µs** of avoided host work — two reuses to break even — and the measured hook costs **2.318 ms per step for 6 promotions ≈ 386 µs/swap** (`T31-strata-transfer.md:379`), ~9× the work it avoids |
| 4 | **inline in decode**: 2.318 ms of a 25.9 ms step = **~9 % of the token** | off the critical path, exposed cost ≈ 0 | the adaptation pays for itself only if the speed-up exceeds its own overhead |
| 5 | **`EXL3_MOE_CPU_PROMOTE=0` in the launcher** turns off *admission* (`spares = 0` → `_split_promote_step` returns, `:1482-1485`) — **but the sweep is live**: `EXL3_MOE_CPU_SWAP` is unset, so `_split_dynamic` is true (`:1354`), and the load line says `(worker, dynamic)`. 64 swaps per 128 steps (`:141`, `:1486`), exchange-in-place | always on | the box **does** adapt; §3's defect is the promote path, not adaptation as a whole. See the correction below |

Item 1 is the design defect that explains the negative result: a promoter whose slots come out of the tier it
is meant to improve is a negative-sum trade at every budget. Item 3 explains why the cadence cannot rescue it.
Item 5 explains why the served box sees none of it.

## 4. Why the arm that "closed" it does not close the question

- `t29-C1-pin-unset-20261103-221302` reached **72.35 %** coverage (against 31.76 %) and bought no decode — but
  it ran at 16 spares (115 static residents, §3.1), at the shipped 6/step budget with 0.98 decay, with the
  swap inline, on the pre-drafter path, at a 131 072 ladder.
- `t39` then measured pin + promoter **together** on the drafting path and withdrew it (`Coding −4.28 %`,
  weighted primary 0.9997) — that is a verdict on the shipped shape, not on the mechanism.
- The cadence knobs that would make it Strata-shaped (`EXL3_MOE_CPU_PROMOTE_EVERY`, `_DECAY`) appear in
  **0 of 232** rows of `passes.csv`. They have never been run.
- The per-layer uniformity has never been varied. Strata measured the *reverse* move on its own engine and it
  cost **−15.5 %** of throughput (hit rate 94.1 → 88.6 %, `/tmp/opencode/strata-vs-exllama-research.md:117`)
  — evidence that the topology is worth a double-digit percentage, and that we are on the wrong side of it.

## 5. The measurement hole

Coverage has not been read since **2026-11-03**, and never in the served regime: never at 262 144, never with
DFlash2 attached, never with the pin on. A 5-row verify window multiplies `(row, expert)` evaluations by 5
while committed tokens only multiply by ~3.05, so a wider window should **lower** this coverage measure.

**Pre-registered prediction** for the one cheap measurement (one pass, `EXL3_MOE_TIER_STATS=1`, served config):
pooled coverage **26–32 %**, i.e. at or below 31.76 %. A reading above 35 % means either the instrument or the
placement differs in the served config and the ledger needs re-deriving.

Two source disagreements stay open, neither changes the direction:

- The prior research note derives the fork's coverage as **~41 %** from bytes; the instrument says **31.76 %**.
  Prefer the instrument (it counts evaluations). Note that **42.9 %** is exactly the coverage at which the
  measured 1.645× closes with no second mechanism — the note's number sits there, which is why the
  disagreement matters.
- Strata's per-token host bytes have two derivations differing 3.4× in the prior note; the accounting uses the
  larger, which is the one that makes the ratios close.

## 6. So: what is the strategy we are failing to implement?

Not "pin the arena" (shipped, +23–25 % prefill, 0 % decode). Not "port the hot profile" (ported, at parity).
Not "buy more VRAM" (worth 4 % for 21 experts, and refused at 262 k). The strategy is:

> **One global expert pool, re-ranked across layers at a cadence measured in tens of swaps per round, off the
> critical path, with the pool's total fixed.**

The fork has every component of that sentence except the shape, and it has never run the shape. That is the
transfer that has not happened.

**Cost at 262 k: zero VRAM.** A global pool redistributes the same 6,288 slots across 48 layers instead of
giving each 131; it needs a per-layer `first` vector (`:1314`, `:1456`) and a swap budget that does not remove
slots from the static set. It is a code change, not a memory purchase — which is why it is the only remaining
lever that is compatible with the mandatory context.

## 7. Open, in order of what would settle most

1. Read coverage in the served regime (prediction above). One pass.
2. If it is at or below 31.76 %: build the global-pool shape with the budget/cadence knobs, and pre-register
   its prediction (coverage → ≥50 %, decode ≥70 tok/s at 130 816) before running it.
3. Measure the swap cost directly (`73 µs` estimated vs `386 µs` derived) — if the real cost is the derived
   one, no cadence can pay and the exchange must move off the critical path first.

Methods: two independent read-only analyses (`/tmp/opencode/strata-decode-path-analysis.md`,
`/tmp/opencode/fork-strata-gap-ledger.md`), each checked against the box's load line, the profile file on disk
and the source it cites. The one claim that did not survive checking — identity placement — is corrected in §2.

## Correction, 2026-11-08

§3 item 5 originally said the shipped box has "no adaptive placement at all". That is true of the **promote
path** only. `EXL3_MOE_CPU_SWAP` is unset in the launcher (only `PROMOTE=0` is set), so `_split_dynamic` is
true (`block_sparse_mlp_cpu.py:1354`) and the **sweep** runs: 64 swaps per 128 steps (`:141`, `:1486`),
exchange-in-place into the coldest resident's slot (`_split_swap_experts`, `:1543-1613`). The box's own load
line confirms it — `CPU split experts (worker, dynamic)` (`:1452`).

The diagnosis is unchanged in direction but sharper in what to test next: the exchange protocol this report
asked for **already ships**, and coverage still sits at 31.76 %. So the open question is not "build an
exchange" but **"does the sweep ever actually swap, and if it does, does 64 swaps/128 steps move coverage?"**
Two candidate reasons it might be inert: the absolute floor `8 × hist.sum()/num_experts` (`:1516-1522`) may
never clear in a short accumulation window, and the generator's idle hook may not fire often enough under
tabbyAPI traffic (`generator.py:1199-1203`). Both are measurable with `EXL3_MOE_CPU_SWAP_DEBUG=1`, which
prints the swap count per sweep and is not set today.

`wiki/plans/2026-11-08-pool-shape-design.md` §0 is the source of this correction; the design was reviewed
against the launcher and the load line before being accepted.
