# Task 5 plan: sample-in-graph / terminal-sync removal (handoff) — SERVER PATH ONLY

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
- Sampler object: `DefaultSampler` (`job.py:166-168`); read its
  `sample()` for multinomial/temp/top-k paths (RNG in-graph
  needs static Philox state — note as risk, do not solve yet).

## 1. Read this first (no code until done)

1. `generator.py:650-700` and `:770-800` (the two sample sites +
   surrounding orchestration: what consumes the token on host
   and when).
2. `generator.py:1110-1140` (the full synchronize: what breaks
   if it goes away).
3. `job.py:600-640` (per-step token extraction + stop checks).
4. Task-6 plan §8 (in-process 19.27ms vs server 21ms: ~1.7ms is
   loop overhead ABOVE the model — your prize pool ceiling).

## 2. Phase 0 — premise check + STOP gate (measure, no code)

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

No sampler-math changes (same distribution, same seed ->
same tokens), no stop-string behavior change, no prefill edits.
Hands-off everything outside scope; `git status` + `git diff
--stat` before every commit, explicit per-file `git add`.
Spikes untracked, atomic commits, ledger lines on landing.

## 6. Done means

Server-harness tok/s at 64k ctx (greedy) improves by the §2
measured prize (±20%) with token-identity twin + PARITY green,
ledger entry, pushed — or documented SKIP with the §2 bill.
Microkld tg is not expected to move; do not chase it here.
