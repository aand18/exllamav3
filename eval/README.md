# eval/ helper manifest (2026-10-05, session knowledge)

Status meanings: **LIVE** = backs current ledger numbers or protocol,
use it. **DEAD (recorded)** = superseded/STOPped, numbers already in
`doc/kvarn-4090.md` or a plan — do not re-run for decisions, do not
delete (evidence chain). **UNKNOWN** = status unclear, check the cited
plan section before trusting it.

## LIVE harnesses (back the ledger)

| file | backs |
|---|---|
| `kvarn_microkld.py` | tg/pp/KLD, every table (protocol v3) |
| `kvarn_needle.py` | needle gates (27B + Flash-Next 4/4 @131k) |
| `kvarn_torture.py` | torture gate |
| `kvarn_prof_kineto.py` | Kineto bills (serve/combine splits) |
| `kvarn_profile_decode.py` | decode profile |
| `kvarn_compile_probe.py` | triton compile checks |
| `smi_guard.py` | VRAM watchdog (protocol, every box run) |
| `perf.py`, `spec_decode.py` | tabbyAPI battery (their harness) |
| `ppl.py`, `qbench*.py`, `bbeh_mini.py`, `humaneval.py`, `ifbench.py`, `mmlu.py`, `diversity.py`, `longctx.py`, `model_diff*.py`, `prequant_test.py` | benches / upstream utils |
| `_probe_mem64.py`, `_probe_vram.py`, `_probe_prefill.py`, `_probe_prefill_fp16.py`, `_probe_serve_delta.py`, `_probe_eref_delta.py`, `_probe_step.py`, `_probe_arm.py`, `_probe_seal_compile.py`, `_probe_geom.py` | committed diagnostics — probe before theorizing |
| `_spike_graph_wrap.py` | LIVE: imported by `test_graph_wrap_matches_eager` (do not delete/move) |
| `_spike7_coal.py` | LIVE reference: production serve kernel is a verbatim copy (see comment at `kvarn_triton.py` promoted-serve header); delete only when dispatch owns the path |
| `_spike9_ab.py` | LIVE method reference: single-process interleaved A/B protocol (the only quotable protocol on this box) |
| `_spike23_q5.py` | LIVE: q5-vs-kvarn KLD harness (retained artifact; 27B re-measure pending) |
| `_r154_b0_compat.py`, `_r154_b0_sigs.py` | DONE record: 26-symbol backend compat for the 1.5.4 upgrade |

## DEAD (recorded) — do not re-run for decisions

| file | superseded by |
|---|---|
| `_spike2_online.py`, `_spike3_online.py`, `_spike4_online.py`, `_spike5_single.py`, `_spike6_mma.py` | early serve exploration → spike7 → production |
| `_spike8_layer.py`, `_spike8_split.py` (+`.bat`) | task-6 STOP (+0.21%, plan §8) |
| `_spike9_serve_bill.py`, `_spike9_dbg.py`, `_spike9_dbg2.py` | task-7 bill (numbers in plan §8) |
| `_spike10_tailmerge.py` (+bats) | task-4 STOP (−4.5% end-to-end) |
| `_spike11_sampler_bill.py` | task-5 STOP (ceiling 0.25ms, plan §7) |
| `_spike13_loadtest.py`, `_spike14kld.bat`, `_spike15_kldctl.py`, `_spike16_collect.py` | Flash-Next bring-up numbers (ledger FN section) |
| `_spike17needle.bat` | FN needle 4/4 (box logs `fn_needle_*`) |
| `_spike18_kvarnonly.py`, `_spike19_kvarndet.py`, `_spike20_bisect.py` | FN grid/determinant/bisect (ledger FN section) |
| `_spike21_kvtime.py` | ABANDONED: crashes under graph capture (`0xC0000409`) — kept as constraint record, use `_spike22_getkvtime.py` pattern instead |
| `_spike22_getkvtime.py` | get_kv bill (task-7/FN reports) |
| `_dbg_*.py` (all 13) | early debug, long superseded |
| run bats (`_a15_*`, `_a2*`, `_b0*`, `_b1*`, `_b2*`, `_r154*`, `_reverify*`, `_envprobe*`, `_extprobe*`, `_fn_*`, `_spike*bat`, `run_*.bat`) | one-shot runners; numbers archived in box logs where noted |

## UNKNOWN — check before use

| file | note |
|---|---|
| `_spike12_audit.py` | Flash-Next audit; see bring-up plan before trusting |
| `_spike23_table.py` | q5 table helper; status unclear, see q5 note in ledger |

Rule for future spikes: add one line here (status + backing pointer)
in the same commit that records its numbers, or it defaults to UNKNOWN.
