# KV-cache quantisation benchmark — quality at depth + VRAM frontier at 256k

Goal: decide the serving `cache_mode` for Qwen3.8-Flash-Next-exl3 3.05bpw on
the 4090 box for the operator's real workload — agentic coding, max 256k ctx,
opencode-driven (xhigh thinking). Production ships `cache_mode: 2,2`
@ `cache_size: 262144`; its **quality has never been measured** — the
flash-knobs plan shipped it on perf evidence only ("cache_mode quality was
never assessed; KLD owns quality", `flash-knobs-benchmark.md` §3.1). This plan
owns that gap.

**KVaRN is out of scope by operator decision (pinned memory 2026-10-07: "too
slow, minimal VRAM gain for this model").** Bare symmetric pairs only; never a
`kvarn*` preset; `-kvt/-kvt_type/-kvsk/-kvsv` inert. KVaRN's measured quality
ledger does NOT transfer to the plain-`Cache` path (different dequant/serve
path, obs #101) — plain-path arms start from zero evidence.

## A. Why this is not a re-run of the knob battery

The arch changes what KV quantisation can damage:

- Only the **12 QSA layers** hold KV pages; the 36 GDN layers are recurrent
  state and carry no KV. Quantisation touches a minority of the net.
- The QSA indexer caps every query at **512 blocks × 4 = 2048 attended
  tokens** (`qsa_indexer.py`, obs #140). So KV quant cannot "blur the whole
  context" — it corrupts either (a) the *retrieval selection* (indexer picks
  wrong blocks) or (b) the *values inside the 2048-token window the query
  actually reads*. Both failure modes are retrieval errors at depth.
- The **indexer planes stay fp16** (~320 B/token/layer, obs #56) and are
  quant-invariant. Consequence for method: **the index is clean, so the
  quantised model self-reports the damage** — forced-choice MC answers with
  the correct span inside the 2048-token window are a valid same-item
  reference, because retrieval itself is not what's perturbed.
- Attn KV cost is only 12 layers × 2 KV heads × 256 dim → fp16 ≈ 24 KB/token,
  ≈ 6.3 GB at 262144. The fp16 *indexer* floor (~3.8 KB/token) is paid by
  every arm, so quant savings are smaller than they look.

**The one structural property that makes a true reference possible:**
attention is causal, so tokens up to depth *L* produce byte-identical logits
and KV regardless of what is written later. Two runs with the same prefix and
different filler beyond *L* share KV through position *L*. A run can therefore
carry **one shared cold-prefill reference prefix for all depths**, with depth
as an in-boot sweep variable instead of a boot variable. This kills the
boot-count wall and is common-mode against machine state — the weakness that
burned the 250k measurements before.

## B. Metrics — chosen for what they can discriminate

| id | probe | discriminates |
|---|---|---|
| P1 | **Depth ladder** (new harness, §D1) | the *shape* of recall-vs-depth degradation: onset point, slope, plateau. Primary quality result. |
| P2 | **Needle ≥7/8** (§D2) | clean pass/fail gate on the same mechanism at arm max length; a **same-pair** diff vs reference arm, not a score. |
| P3 | **Production-shaped replay** (§D3) | whether *this* workload (agentic_code_10, 11 tools, 11–16k) is affected: tool-call/tool-result reuse accuracy across turns, turn wallSec vs noise band. |
| P4 | **FP8 control** (§B) | separates "bits are insufficient" from "this dequant path is broken on this arch". |
| P5 | **KLD vs fp16 KV, 16k** (optional, §D4) | attributes a P1/P2 delta to distributional drift vs behavioural noise; gated by `wiki/patterns/kld-median-noise-floor.md` (same-top + mean/max band, never median digits). |

Not used as primary: free-gen divergence (thinking-length variance swamps
signal), KLD as headline (1e-6 prints are noise; gate pattern), prose-summarise
probes (wrong failure mode for a retrieval-capped arch).

### P1 — the depth ladder (new; primary)

Synthetic KV-depth ladder built on the existing `_kb_mklongctx.py` recycler:

- Build ONE real conversation prefix (~2–4k tok: tools + agentic preamble,
  ~10k template overhead included) + N needle trials. Each trial: a 1–2k-token
  verbatim excerpt (code blocks, config lines, tool-result JSON — *code-flavoured*
  needles, because the operator's KV content is code) planted at depth
  `D ∈ {8k, 16k, 32k, 65k, 131k, 250k}`, then a **forced-choice question**
  naming the span ("which of these 4 values is the timeout set in config X
  above?").
- **One boot per arm** streams depths ascending: depth *D* query is appended to
  the shared prefix → warm prefill, no cache eviction, and the reference is
  the same arm's own answer at 8k plus the fp16 arm's answer at the same depth.
- Forced choice → greedy, deterministic, auto-scored. Record per (arm, depth):
  correct?, latency-vs-band (the `wallSec` signal already validated in
  flash-knobs §0.8), log-verified position of the span, and **indexer-window
  coverage**: is the span's block inside the selected 512 blocks at the query
  position? A miss with the span *outside* the window is a retrieval-selection
  failure; miss with span *inside* window = value corruption. The harness must
  log selected block ids (probe before theorising — if block ids are not
  exposed, approximate via span-position sweep within one window).
- One cold prefill per arm (~2 min at 250k, log-verified) + ~30 s of queries:
  P1+P2 cost ~3 boots for the whole ladder.

### P2 — needle, at the gate practice

`eval/kvarn_needle.py` with **N ≥ 8 trials per length**: at p ≈ 0.85 (the
plausible degraded-arm regime) a 4-trial test has a ~40% false-pass rate;
8/8 vs 4/4 discriminates at α ≈ 0.43, 8/8 vs 6/8 is already suggestive, and
12/12-vs-8/8 closes it. Gates: arm must match ref-fp16 at 131k (ref's standing
4/4 becomes 8/8 under the new protocol) and hold ≥7/8 at 200k+. Report per-arm
per-length pass counts, not scores.

### P4 — FP8 control (new arm)

`CacheLayer_quant` is 2–8 bit integer (assert at `cache/quant.py:30`); FP8 KV
is a different quantiser on a different kernel path. Check whether
tabbyAPI's exllamav3 backend exposes it (`fp8_kv`/`quantized_kv`); if it does,
run it as the ceiling control: if FP8 also degrades recall, suspicion moves to
the dequant/serve path rather than bit width — the exact lesson of obs #101
(harness/path artifacts masquerade as model behaviour).

## C. Arms and the VRAM frontier

| id | arm | role |
|---|---|---|
| `ref-fp16` | FP16, cache_size = max bootable (est. ≤131k) | quality ceiling; NOT a 256k candidate |
| `cur-22` | `2,2` @ 262144 | production under trial |
| `q33`/`q44`/`q55`/`q66`/`q88` | ladder @ 262144; descend cache_size (224→192→160→128k) where boot fails | bits-vs-context frontier |
| `fp8` | if backend-supported | mechanism control |

FP16 arm economics changed 2026-10-07 (operator): `start_tuned.ps1` now ships
`cpu_moe_split_experts: 380` + `EXL3_MOE_CPU_THREADS=16` (the battery winner,
−724 MB vs the old mcl38 regime) and `vision_offload: true`, freeing ~1.5 GB
over the regime the old ladder booted in. **Re-measure bootability** — do not
carry the old "won't boot" verdicts; and remember the live server sits
+1.7–2.5 GB above the offline harness, so offline "fits" proves nothing.
FP16@262144 (~6.3 GB KV) is still expected to miss on a 64 GB box (RAM binds
at load, flash-knobs §0.9 #1) — but measure, don't lore.

Output of this phase: `max_bootable_tokens` per arm at **both** memory regimes
— that table IS the decision material, since the frontier question ("how many
bits for how much context") is the operator's call, not the harness's.

## D. Protocols

**D1 ladder harness** — extend `eval/_kb_mklongctx.py` (`--ladder` mode):
emit shared-prefix + depth-parameterised queries; emit scoring key; validate
all variants against the two-term prompt-size law
(`0.2383×text + 294.4×n_msg + 9110`) and reject the batch if any variant
exceeds `cache_size − 256`. Score automatically; write per-(arm,depth) rows.

**D3 production replay** — reuse the §0.8 sustained-replay protocol
(`agentic_code_10.json`, turn i = `messages[0..i]`), temperature 0, one boot
per arm interleaved with reference; primary estimator = median per-turn
wallSec ratio; correctness = tool-call name+args + final-answer patch vs
`ref-fp16`'s run of the same turns. Skip arms with >1 tool-call divergence at
16k unless screening is the point.

**D5 determinism — the open pre-flight (NOT yet a rule):**
`EXL3_DSA_QC_STAGE=0` pins the online-dequant attention path (deterministic,
no threshold cliff at ~1M entries) but forfeits the gather-once staging win
(~3.9× faster attention at 16k; ~95% of 250k wall-clock is prefill).
**Unmeasured how much of that applies to the plain `Cache` path** — the 3.9×
figure came from the kvarn-armed harness. Pre-flight #1 therefore measures it
on the plain path (same prompts, `EXL3_DSA_QC_STAGE` 0 vs 1, boot-interleaved,
wallSec + logits fingerprint). Quality runs: arm internal A/B only (each arm's
answers vs its own fp16 control), never cross-stage ratios. Perf/timing rows
use the default (staged) path — production default is what ships.

All D phases inherit verbatim: interleaved reference (§0.8.1), ≥4 distinct
variants per boot at long context, "within-config spread > between-config
spread ⇒ uninformative", ≤3 pp = noise, no ratio before the last arm lands,
`config.yml` .bak + md5 invariant checked only when idle, startup-log value
verification, `start_tuned.ps1` always, restart announcements.

## E. Run order (boot-cheap first)

1. Pre-flight: plain-path `EXL3_DSA_QC_STAGE` cost; FP8 support check; block-id
   logging availability probe. (3 boots, all reusable knowledge.)
2. Frontier: bootability at 262144 for `3,3→8,8` (+FP8), then cache_size
   descent; boot-time + VRAM + RAM rows.
3. `ref-fp16` @ max bootable: ladder P1 depths ≤ its ceiling + P2 at 32k/65k.
4. `cur-22` full: ladder to 250k, P2 at 131k/200k+, P3 replay, P5 at 16k.
5. Survivors per arm quality-vs-length; P4 control if budget allows.
6. Report: frontier table + per-arm quality-vs-depth curves + proposed
   `config.yml` diff (never committed) + verdict only where evidence supports
   it.

## F. Known costs, stated

- Ladder depth is capped by `cache_size`: at 262144, max usable D ≈ 250k after
  the ~12k tools/preamble overhead + generation headroom (flash-knobs §0.7
  calibration, verified to 0.07%).
- 250k fast/slow machine state is environmental and unidentified; match state
  within boot, never across (flash-knobs §0.8.2).
- P1 scoring is auto but the *probe author* owns validity: spans must be
  verbatim-unique, outside the trailing 2048-token window at query time, and
  not duplicated by the recycler's `[pass N]` recycling.
- First application of the depth-ladder protocol — validate against `2,2`
  (expected: near-fp16 if quant is innocent) before trusting any other arm.