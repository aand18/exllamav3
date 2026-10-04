# Subagent report: merge-readiness audit (2026-10-02)

## Provenance
RECOVERED from chat output (pre-convention dispatch: no report file
was written; this is the complete returned text, lightly reformatted
for markdown — content verbatim). No upstream PR exists yet; this
audit is advisory for when one is planned.

## Original report (verbatim)

**1. Per-commit verdict**

* `6dda3ec` Qh-view -- **as-is**. 1-line, no env/print, true one-cut.
* `bf91130` host-desync -- **as-is**. `EXL3_KVARN_TRITON_PARITY`
  opt-in only, safe default off. No hot-path print.
* `277efe8` store trims -- **as-is**. No env, no instrumentation.
* `7f6fb33` dispatch trims -- **as-is**. Persistent `_out/_bufs=None`
  defaults = legacy. Only touches parity-gate.
* `17373a9` status threading -- **as-is**. None-safe plumbing, no
  behavior change.
* `476eb94` seal-direct -- **after cleanup**. Correct + twin, but
  introduces `EXL3_KVARN_FASTSTORE` default ON (`kvarn.py:402`).
  Safe fallback (`=0`->legacy) but ON-by-default is opt-out;
  upstream will demand OFF-by-default or perf proof. No print/PTIME
  in this cut.
* `3ce676f` 31+1 split -- **after cleanup**. Good rev-resolved
  twins, env save/restore correct. Keeps `_ptime_count(
  "decline_pressure"/f"fast_E{E}")` -- gated (`kvarn.py:415-422`,
  zero-cost off) so keepable, but squash string-cardinality
  (`fast_E{E}`).
* `3647f54` hierarchical serve -- **after cleanup**. Must delete
  `print(grouped serve: maxabs...)`; replace magic `64/128` with
  named consts (`kvarn_triton.py:1822`
  `groups=gc if gc<=64 else min(gc,128)`). Comment cites "144 SMs"
  -- HW-specific, generalize. Twin good.
* `43faa21` tail_gather bufs -- **after cleanup**. Prod change OK
  (`dispatch.py:541` `tailR_*` PTIMES-gated). Mixed-concern: same
  commit touches `doc/graphs-v2-plan.md` +
  `eval/_spike_graph_full.py:44` (`print GATHER-BUFS: ok`). Split
  prod vs doc/spike; drop print.
* `f6d6b5a`+follow-ups graphs branch -- **needs-rework**.
  `f6d6b5a` alone (default OFF, eager fallback) was mergeable;
  follow-ups `fc3f24c/bce5072/b3aeb71` flip `EXL3_KVARN_GRAPH`
  default ON (`dispatch.py:245` default `"1"`, stale comment `:184`
  still says "default off"). Ships `print`+traceback in hot path
  (`dispatch.py:334,338,340,609,611`), `DEBUG_HASH` prints
  (`kvarn_triton.py:1846,1849,1876`), engagement
  `g_captured/g_capfail/tailR_*`. Squash to one commit, default
  OFF, logs->`logging.debug`.

Env added by listed cuts: `FASTSTORE=1 ON` risky, `GRAPH=1 ON`
risky, `PARITY=0 OFF` safe, `PTIMES=0 OFF` safe, `DEBUG_HASH`
opt-in safe. Dead: `_try_kvarn_online_decode`
(`dispatch.py:402`, unwired), `kvarn_triton_online_decode` vs
`serve_online_s7` duplicate bodies (`kvarn_triton.py:1464,1522,1781`
"spike7->production"). No V1 partials left. Tests restore env
correctly except `test_kvarn_triton.py:567-573` assumes unset
(fails if CI exports var). Stale: `dispatch.py:538` wobble,
MMA comments `:1663`, spike7 refs to `eval/_spike7_coal` not
shipped.

**2. Hygiene:** rebaseable, but squash graphs v2 (4->1), split
`43faa21` doc/spike out. Do NOT ship untracked:
`eval/_probe_geom.py`, `doc/perf-strats.md`, `wiki/reports/2026-10-02-*`.
Target stack: `bf91130` -> `6dda3ec` -> `277efe8` -> `476eb94`
(default OFF) -> `3ce676f` -> `3647f54` (consts, no print) ->
`7f6fb33` -> `17373a9` -> `43faa21`-prod-only ->
graphs-v2-squashed-OFF.

**3. Top 5 prep (do not do):** 1. Graphs default OFF + strip
prints/traces `dispatch.py:334-340,609-611` + gate engagement --
2h. 2. `FASTSTORE/DEFER_SEAL` defaults OFF (`kvarn.py:391,402`) +
doc fallback -- 1h. 3. Remove hierarchical print, const-ify 64/128
(`kvarn_triton.py:1822`) -- 30m. 4. Delete/split spike artifacts
(`kvarn_triton.py:1522-1523`, `eval/_spike*`,
`eval/_probe_geom.py`) + dead `online_decode` -- 1h. 5. Fix
env-sensitive test (`test_kvarn_triton.py:567`) + add SWA/multi-
batch twin -- 2h.

**4. Embarrassments:** `print+format_exc(limit=12)` in prod
fallback; `DEBUG_HASH` f-strings evaluated per-serve; magic
`31+1/128/64/CPG`; undocumented invariant "CPG=1 == old path";
`assert free.numel()` staging/exact overflow (`kvarn.py:1279,1441`)
fires in prod on 128k+; zero coverage SWA/multi-batch/128k for
hierarchical/graph paths -- likely perf cliff (128 CTAs vs 144 SMs
assumption).

## Coordinator disposition (2026-10-02)

Filed, not actioned (no PR exists; prep is premature). Two
disagreements recorded: (1) defaults stay ON -- the gains exist
only with them on (pp +30%, tg +16-27%), and gate evidence
(twins + KLD + needle + review) IS the perf proof the audit asks
for; flipping OFF surrenders the branch's purpose. Revisit if a PR
opens. (2) `wiki/` + `doc/perf-strats.md` + eval spikes stay on
the fork (house process artifacts, never proposed for upstream).
Agreed items queued for the eventual prep pass: const-ify 64/128,
drop spike `print`s, fix env-sensitive test, dead-code sweep.
