# T61 — the served drafter is not the measured one: acceptance −0.18, speed unresolved, 0.75 GB freed

Pre-registered in `T61-drafter-regime-gap.md`. Basis: 131072, `spec_decode`, production regime
(split 381, threads 16, Q4 `4,4`, cache 40960 for drafting), static placement, pin on,
`-ndt 4 -dds`.

## Runs

| arm | run id | draft tok/s | accepted (mean of 8 cats) | VRAM floor |
|---|---|---:|---:|---:|
| BF16 drafter | `t61A-bf16-131k-20261110-065833` | 57.10 | 2.056 | 2687 MB |
| exl3 drafter | `t61B-exl3-131k-20261110-071514` | **59.04** | 1.906 | 3627 MB |
| exl3 drafter | `t61C-exl3-131k-20261110-073132` | 57.72 | 1.860 | 3415 MB |
| BF16 drafter | `t61D-bf16-131k-20261110-074755` | 57.02 | 2.078 | 2687 MB |

Arms differ **only** in `-dm`: `D:\llms\Qwen3.8-Flash-Next-DFlash2` (1.2 GB, BF16) versus
`D:\llms\Qwen3.8-Flash-Next-DFlash2-exl3` (472 MB, 4-bit exl3 `head_bits 6`). Mean drafted tok/s:
BF16 **57.06**, exl3 **58.38** (**+2.31 %**).

## Verdict against the pre-registered gate

The gate was: acceptance within 0.05 **and** corpus ≥ 58 tok/s.

- **Acceptance: FAIL, and it is resolved.** 2.067 → 1.883 accepted drafts (**−0.184**), over
  three times the 0.05 tolerance. It is category-dependent, not noise: Agentic-curl 2.61/2.73 →
  2.21/2.16, Translation 1.26/1.27 → 1.02/1.02, while Trivial repetition rises 2.85/2.62 →
  2.99/3.04 and Agentic-code holds 2.85/2.80 → 2.87/2.81.
- **Speed: unresolved.** The +2.31 % group difference is exactly the size of the exl3 pair's own
  spread (59.04 vs 57.72 = 2.28 %); the BF16 pair is tight (57.10 vs 57.02 = 0.14 %). One
  category moves 15 % between the two BF16 arms (Creative-reasoning 40.14 vs 46.39), so the
  corpus mean is steadier than its parts.

**So the exl3 drafter is not a demonstrated speed win, and it costs 0.18 accepted drafts.**

## What that means for the record (the reason this task existed)

The exl3 drafter is **already deployed** (`config.yml:352`). Therefore:

1. **Every campaign number since 2026-11-02 measured a drafter production does not serve.** The
   55–57 tok/s baseline is a BF16-drafter figure; the served engine trades ~2 % of draft
   acceptance for speed inside the measurement's resolution.
2. **The served engine sits closer to the drafting break-even than the corpus claims.** Corpus
   acceptance is 1.88–2.07 of 4; the live log's 39.6 % (1.58 of 4) was attributed to workload
   alone, and part of that gap is now explained by the drafter rather than the traffic.
3. **The exl3 drafter frees 728–940 MB of VRAM at the same split** (floor 2687 → 3415/3627 MB) —
   the measured counterpart of the 1.2 GB → 472 MB file sizes. At the T14 coefficient near the
   floor that is roughly **6–9 GPU experts** production is not using.

## Split re-ladder (T61 #2): not run, and why

The pre-registered condition for #2 was "only if #1 holds". #1 failed on acceptance, so #2 was
not launched — moving the goalpost after seeing the result is exactly the failure this campaign
has been auditing out.

The open question it would have answered is nonetheless real and narrower: **production runs
split 381 with 0.73–0.94 GB spare.** Whether serving at a lower split (more GPU experts) is
faster with the exl3 drafter is a *production config change*, and per this repo's rules it
needs the operator before a measurement is planned, because it changes the live split rather
than a measurement arm.

## Recommendation

- **Do not change the served drafter.** It is already exl3; this task removes the option of
  "converting it" as a future win and records its real cost (acceptance −0.18).
- **Re-found or annotate the record.** Any report quoting 55–57 tok/s should name the BF16
  drafter; a future campaign that wants to predict production should run `-dm …-exl3`.
- **The one lever this opens** is the split: ~0.75 GB is idle in production. That is one
  operator-approved arm at 262144 (does the static floor move from 381 to ~372?), and it is the
  only remaining change with a plausible few-percent effect that needs no build.
