# Subagent report: async-seal correctness audit (2026-10-02)

## Provenance
RECOVERED from chat output (pre-convention dispatch: no report file
was written; this is the complete returned text, verbatim).
Session `ses_f05ae0432ffedQ7XymC47Lv2v8`. Intermediate reasoning not
preserved. Outcome: premise WEAKENED (not dead) -> refined design
(rows-stash instead of staging-pin; see coordinator note below).

## Original report (verbatim)

Premise survives only weakened: settled-group records reads are
unaffected by lag, and in-flight groups divert via the `sealed` flag
to staging -- iff staging is pinned. Literal "nothing reads records
in prefill" is false.

**1. Records readers (production only)**

Prefill-reachable:
- `kvarn.py:2682-2694` `_refresh_into` sealed branch ->
  `2618-2656` `_dequant_groups_batched[_torch]`
  (`kvarn_triton.py:818-881` same call), via `2810` get_kv image
  dirty sweep, `2866` legacy path, `qsa_indexer.py:708` get_kv.
  Every chunk's forward reads all previously sealed body groups.
- `kvarn.py:2998` copy_page sealed copy `source.records[gf]`;
  `3050` partial via `_staging_from_records(gf, source.records)`
  (`job.py:1289`, `model_tp_fn.py:384` prompt-cache share can
  interleave prefill).
- Pressure writers racing side-stream records writes (not readers,
  must fence): `kvarn.py:1268` `_stage_slot->_seal_full_groups(None)`,
  `1393` `_reclaim_stage_slot->_seal_group`, `2180-2181` end-of-call
  `_seal_spilled_batched`/`_seal_full_groups`.

Decode-only:
- `kvarn_triton.py:1804,1844` + kernel `1532-1869`
  `kvarn_triton_online_serve` (via `dispatch.py:517-520`), gated by
  `sealed` mask + sticky flag; `kvarn.py:1549-1641`
  `kvarn_online_tail` staging fallback (reads exact/staging, not
  records).

**2. Verdicts**

- Inter-chunk `_refresh_into`: safe-under-lag IFF staging slots of
  pending groups are pinned -- `sealed=False` diverts in-flight
  groups to the staging branch (`2722-2730`), serving exact fp16
  (note: numerics diverge from eager dequant; parity/KLD must
  tolerate). UNSAFE if slots released early: `rev<0` path clamps +
  zero-fills (`2724-2727`) -> 128 zero keys into SDPA. UNSAFE on
  pressure: 4 stage slots vs 32 groups/4k prompt forces sync seal
  racing the side stream.
- copy_page: UNSAFE -- `sealed` check (`2995`) vs records copy
  (`2998`) TOCTOU against side-stream completion; staging copy
  (`3042`) races seal's staging release -> zeros/partial rows +
  flag mismatch; partial rebuild (`3050`) reads zero payload +
  stale scales.
- Imageless serve: UNSAFE without fence (open body rows trip
  sticky flag or read zero payload/stale scales -> wrong logits);
  safe after fence (invariant: sealed mask + records settled; tail
  partitioned by `exrev>=0`, `dispatch.py:533-539`).
- `_apply_exact_overlay` (`2521`), `_evict_exact_all` (`2186`):
  safe -- read exact/owner only. `get_tensors`/shape reads: safe
  except relocation needs write fence.

**3. Fence design**

Pin staging `rev` for pending groups (forbid `_stage_release`/
recycle/reset-unseal of pending). One device sync side->default
before: first decode serve (incl. graph capture), any copy_page
(source+dest), any `get_tensors` offload/TP export, any
`_store_rows` that could reset/reuse a pending group (page reuse/
spec overwrite), and inside `_stage_slot`/`_reclaim` pressure path
(pending cap ~= 4 slots; overflow forces inline sync, bounding
overlap to ~512 tokens). Forbid in flight: staging release of
pending groups, `_seal_full_groups(None)` scans, copy/offload
touching pending ranges, same-base reuse resets. No fence needed
before inter-chunk get_kv (divert covers it) or before
evict/overlay scans.

## Coordinator refinement (2026-10-02, supersedes staging-pin)

Rows-stash instead of staging-pin: fast-path seals from call rows
(`rk/rv`), so stash `(rk, rv, gs)` refs (+ event) instead of
pinning scarce slots (4 slots cap would bound overlap to ~512
tokens). No staging consumed by fast groups at all; partials go
legacy (sync). Bound in-flight groups (drain on overflow). Only
the records-write goes async; exact/eref/owners stay synchronous.
Fence before: first serve, copy_page, offload/TP export,
reset/reuse of pending groups. End-of-call seals must SKIP pending
groups (else duplicate seal races side stream).
