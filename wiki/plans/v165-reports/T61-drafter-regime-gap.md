# Corrections — the drafting levers (2026-11-10)

Two claims from the last feasibility pass do not survive checking against the box. Both concern
the same subsystem, and one of them hides a measurement/production mismatch that has been in
every campaign since 2026-11-02.

## 1. `-dds` was tested; what is missing is the closing reference run

`-dds` (`dynamic_draft: true`, ceiling `draft_num_tokens: 4`) is **deployed** — production
`config.yml` (`draft_model.dynamic_draft: true`, `draft_num_tokens: 4`) and the measurement
regime (`state.json active.draft.dynamic_draft: true`). Corpus evidence: two arms beat the
reference nearest each by **+1.4 %** and **+3.6 %** (STATE.md "drafting" line, T6 steps 3–5).
What remains is not a first test but the **third run**: the reference itself moves by about the
size of one of the deltas between identical runs, so a two-arm/one-reference shape cannot
separate sign from size. That is why the item is "a third reference makes the pair a true ABBA"
— cost is runs only, no build, no code. Verdict text should say **"close the `-dds` ABBA"**,
never "measure `-dds`".

## 2. The DFlash2 conversion was done on 2026-11-02, and it IS deployed in production

On disk:

| path | size | what |
|---|---|---|
| `D:\llms\Qwen3.8-Flash-Next-DFlash2` | 1.2 GB | BF16 drafter (the source of the conversion) |
| `D:\llms\Qwen3.8-Flash-Next-DFlash2-exl3` | 472 MB | 4-bit exl3 drafter, `num_hidden_layers 5`, `bits 4.0`, `head_bits 6`, mul1 |
| `D:\llms\DFlash2-exl3-work` | — | converter scratch (`args.json`, `ckpt`, `qtensors`), 2026-11-02 20:47 |

Production serves the converted one: `config.yml:352` `draft_model_name:
Qwen3.8-Flash-Next-DFlash2-exl3`. So "DFlash2 conversion" is not a remaining lever — the
ranked list in the feasibility pass was reading a stale `STATE.md` open item.

## The finding underneath: measurement regime ≠ production regime on the drafter

Every campaign arm passes `-dm D:\llms\Qwen3.8-Flash-Next-DFlash2` — the **BF16** drafter — via
`bench/windows/state.json:26`, because that is the T7 DRAFT_WINNER ("DFlash2 BF16, window 4",
frozen 2026-10-30). Production, since 2026-11-02, serves the **472 MB exl3** drafter.

Consequences, stated so nobody has to rediscover them:

- **The measured 55–56 tok/s corpus baseline is a BF16-drafter number.** The served engine has
  ~0.75 GB more VRAM free than any campaign arm assumed. At ~21.9 GB used of 24.5 GB, 0.75 GB is
  roughly **8–9 GPU experts** — exactly the order the split ladder spends ~1 % each on (T14).
- **Nobody has measured the exl3 drafter's own tok/s and acceptance** in the harness. Its
  break-even is the drafter's own: acceptance must stay above ~1.5 accepted of 4 for drafting to
  pay. A 4-bit drafter with `head_bits 6` may trade a little acceptance for the VRAM; that
  trade has never been priced on this box.
- **Split 381 was founded with the BF16 drafter resident.** With the exl3 drafter the static
  floor may sit lower, buying GPU experts that the campaign's own coefficient (~1 %/expert near
  the floor) prices at a few percent — above the 3 % same-session spread only if the floor moves
  by ≥3 experts.

## What this changes in the ranked list

Replace "DFlash2 exl3 conversion (~+1.9 %)" with two measurements that do not exist yet:

1. **Drafter regime reconciliation (no build).** One `spec_decode` pass with
   `-dm ...-DFlash2-exl3` at the current split 381, ABBA against the BF16 reference, reporting
   acceptance and tok/s. Pre-registered pass: acceptance within 0.05 of BF16 **and** corpus ≥58
   tok/s; a drafter that loses acceptance is not shippable regardless of VRAM.
2. **Split re-ladder under the exl3 drafter (no build, `-Split` deltas).** Only if (1) holds:
   find the new static floor at 262144 and price each expert against the same-session reference.
   Pre-registered pass: ≥+2 % decode at 131072 or the floor does not move.

Both are harness-only. Neither needs a wheel, a kernel, or an engine change.

## Evidence base and gate

This report cites no new run: it is a reconciliation of the deployed configuration against the
campaign record, so the post gate reports "cites no run ids" by construction. Evidence used:

- `config.yml:352` (`draft_model_name: Qwen3.8-Flash-Next-DFlash2-exl3`), `config.yml` draft
  block (`dynamic_draft: true`, `draft_num_tokens: 4`) — the served regime.
- `bench/windows/state.json:26` (`draft_model: D:\llms\Qwen3.8-Flash-Next-DFlash2`) and every
  arm's `-dm` in `passes.csv` (`t58A-slots4-131k-20261108-214726`,
  `t58B-slots8-131k-20261108-220339`, `t58C-slots8-131k-20261108-221944`,
  `t58D-slots4-131k-20261108-223615`, and `t60B-autocorr-131k-20261109-181208`) — the measured
  regime, all BF16.
- `D:\llms` directory timestamps and sizes (2026-11-02 20:47, 472 MB / 1.2 GB), and
  `Qwen3.8-Flash-Next-DFlash2-exl3/config.json` (`bits 4.0`, `head_bits 6`, `num_hidden_layers 5`).
- `wiki/plans/v165-STATE.md` — the `-dds` two-arm record (line 165) and the stale DFlash2 open
  item (line 219).

The report is a configuration finding, not a measurement; the two measurements it proposes are
the ones that would carry run ids.
