# CSV data set — Flash-Next knob battery (2026-10-06)

Machine-readable companion to
[`../2026-10-06-flash-next-knobs-consolidated.md`](../2026-10-06-flash-next-knobs-consolidated.md)
(readable summary) and
[`../2026-10-06-flash-next-knob-battery.md`](../2026-10-06-flash-next-knob-battery.md)
(audit trail with the full correction history).

Regenerate with `python3 eval/_kb_mkcsv.py`. Phase A/B/C rows are read live from
the run logs in `tabbyAPI/logs/kb/*.jsonl`; the remaining tables are literals in
that script, which is deliberate — it keeps them diffable and reviewable rather
than retyped at analysis time.

## Files

| file | rows | grain | what it answers |
|---|---|---|---|
| `measurements.csv` | 255 | one row per **run**, per phase | what did every arm actually record — VRAM, status, boot, sustained tg/pp |
| `long-context.csv` | 26 | one row per **stage × arm × boot** | how the gain varies with prompt length, and which machine state each boot was in |
| `draft.csv` | 19 | one row per **draft measurement** | the `ndt` ladder, draft acceptance by content category, and the `draft_cache_mode` ladder |
| `decisions.csv` | 24 | one row per **knob** | the verdict table — adopt / keep / reject, with gain, cost, confidence, rationale |
| `retractions.csv` | 11 | one row per **withdrawn claim** | what was asserted, why it was wrong, what corrected it |
| `methodology.csv` | 10 | one row per **protocol rule** | the rule, why it exists, and the specific error it prevented |

## Conventions

- **All numbers are medians** unless a column name says `_min`/`_max`.
- **`confidence`** ∈ `HIGH | MED | LOW | SCREEN`. `HIGH` requires the reference
  arm measured in the **same time window** as the candidate. `SCREEN` = not
  measured to plan depth, recorded so it is not re-tested blindly.
- **`ratio_vs_baseline`** < 1 means the candidate was **faster**. Ratio is
  turn-matched on identical prompt-token counts.
- **`machine_state`** ∈ `normal | fast | slow`. Both arms enter this at 250k;
  ratios are only comparable **within** a state. See `retractions.csv` #8.
- **`source`** marks provenance per row. **`throughput_tier_known`** is `no` for
  Phase A/B and `yes` for Phase C — see gap 1.
- **Silent-failure note:** the first version of the generator shipped with all
  Phase A throughput empty and said so in the README, which read like data loss.
  It was two bugs — an unstripped ANSI regex, and `os.path.exists()` on a Windows
  path from WSL. The generator now prints a `WARNING` listing any `.out` it could
  not open, so this class of failure cannot be silent again.

## Known gaps in the data itself

1. **Phase A throughput is NOT tier-specific — read `throughput_tier_known=no`.**
   The harness writes **one `.out` per arm** (`a29-mcs380.out`), so every tier
   re-run overwrote the previous tier's output. What survives is the *last* run
   for that arm, and `perf.py` does not record its own tier inside the file.
   Consequences:
   - `tg0/pp256/pp4096` are populated (122 of 138 Phase A rows), but the value is
     the arm's last surviving run, **not** the tier named in the `tier` column.
   - The Tier 2 table in the consolidated report (baseline 25.68 / 26.16 / 25.29)
     was transcribed from console output **at run time** and is *not* in the `.out`
     files. It will not match the `tg0` column here, and that is expected — they
     are different measurements of the same arm.
   - **The Tier 2 numbers are therefore the authoritative Phase A throughput**, and
     they exist only in the report, not in this data set.
2. **`measurements.csv` Phase C rows are individual runs, not arm aggregates.**
   Per §0.5, cross-process medians are not valid — aggregate by arm only within a
   single interleaved batch.
3. **`pp_tps_median` is a turn-1 metric in conversation mode.** Later turns are
   near-total cache hits and the >1000-token gate correctly suppresses their
   prefill rate, so `pp median` can be legitimately empty.
4. **Content and tool count are confounded** in the cross-category rows
   (`draft.csv`, `test=acceptance_by_category`): `translate_02` has 0 tools while
   the code/curl files have 11 and 29. The direction of the `2,2` result is solid;
   the prose-vs-code ordering is not.