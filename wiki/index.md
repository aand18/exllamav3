# Wiki (WikiSkill-lite): persistent compounding knowledge

Three layers, following WikiSkill (arXiv:2608.27454, verified):
raw traces (box logs, profiles, `doc/kvarn-4090.md` chronological
record) -> THIS wiki (patterns + impact, never rolled back) ->
skills and process (gated changes, rollback allowed).

## Maintainer contract (periodic subagent pass)

1. Read new `doc/kvarn-4090.md` entries + `skill-impact.md` since
   the last pass (see log below).
2. Consolidate durable findings into `patterns/` (one page per
   finding: problem, rule, evidence with commit refs, scope).
   Chronology stays in the doc; only reusable rules come here.
3. Propose skill/process edits, if any, with validation plan
   (twin/suite/KLD gates). Record every accept AND reject in
   `skill-impact.md` with reason -- rejected interventions must not
   be re-proposed without new evidence.
4. Append the pass to the log below. Never rewrite history.

## Catalog

- `patterns/fancy-index-noop.md` -- indexed-assignment rule.
- `patterns/triton-hot-cache.md` -- protocol v2 (mandatory).
- `patterns/serve-parallelism-bound.md` -- CTA math before traffic.
- `patterns/slot-order-not-contract.md` -- rev maps, not indices.
- `patterns/dont-fuse-mma-into-simt.md` -- arithmetic before code.
- `patterns/cuda-graphs-decode.md` -- replay sublattice, fallback rest.
- `patterns/kld-median-noise-floor.md` -- gate criteria.
- `patterns/benchmark-fp16-first.md` -- mining + determinism rules.
- `skill-impact.md` -- accept/reject audit trail + pass log.
- `reports/` -- full subagent reports (raw layer). Convention:
  thorough-on-disk, terse-in-chat (see `reports/README.md`);
  ctx_index each report so it survives compaction.
