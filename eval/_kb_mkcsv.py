#!/usr/bin/env python3
"""Generate the reusable CSV set for the Flash-Next knob battery.

Reads the Phase A/B/C jsonl run logs and emits flat, self-describing CSVs so the
results can be re-aggregated without re-reading the markdown report. Where a
number only exists in the run log (not recoverable from disk -- the Phase A .out
files were overwritten), it is written from a literal table with a `source`
column marking it, rather than silently retyped.

Outputs (see OUTDIR):
    measurements.csv   one row per phase/arm/tier -- the master fact table
    long-context.csv   one row per stage/arm/boot -- per-length results
    draft.csv          draft-domain: ndt ladder + acceptance by category
    decisions.csv      one row per knob verdict, with confidence and rationale
    retractions.csv    claims made and withdrawn, with the correcting evidence
    methodology.csv    protocol rules and the error each one prevented
"""

import csv
import json
import os
import re
import statistics as st

LOGS = "/mnt/c/Users/yoho/Downloads/tabbyAPI/logs/kb"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "wiki", "reports", "data-2026-10-06-flash-next-knobs")


def load(name):
    path = os.path.join(LOGS, name)
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def rng(v, fmt="{:.1f}"):
    if not v:
        return "", "", ""
    return (fmt.format(st.median(v)), fmt.format(min(v)), fmt.format(max(v)))


def num(x):
    return "" if x is None else x


# ------------------------------------------------------------- measurements
def winpath(p):
    """jsonl stores WINDOWS paths ("C:\\Users\\...") but this script runs in
    WSL, so os.path.exists() on them is always False -- which silently produced
    empty throughput columns instead of an error. Translate explicitly."""
    if p and re.match(r"^[A-Za-z]:\\", p):
        return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")
    return p


MISSING = set()


def parse_perf_out(path):
    """Phase A perf.py stdout, which uses THREE formats depending on tier:
       Tier 0  "Context 0: S=1  26.28 tokens/s"
       Tier 1/2 "Context 0: S=1  26.28 it/s [24.30 - 27.71]"
       plus a "tg0=.. [..] pp256=.. pp4096=.." summary line when present.
       An earlier version of this script only matched the first two of those,
       so the Tier 1/2 finalists read as empty. That was a parser bug, not data
       loss -- see recover note in the README.
    """
    path = winpath(path)
    if not path or not os.path.exists(path):
        MISSING.add(path)
        return {}
    txt = open(path, encoding="utf-8", errors="replace").read()
    # NB: non-raw string. r"\x1b..." would match a literal backslash-x1b and
    # silently strip nothing, which is what made every Phase A row parse empty.
    txt = re.sub("\x1b\\[[0-9;]*m", "", txt)
    out = {}
    m = re.search(r"tg0=([\d.]+)\s*\[([\d.]+)-([\d.]+)\]", txt)
    if m:
        out["tg"] = (m.group(1), m.group(2), m.group(3))
    else:
        m = re.search(
            r"Context\s+0:\s*S=1\s+([\d.]+)\s+"
            r"(?:tokens/s|it/s)(?:\s*\[([\d.]+)\s*-\s*([\d.]+)\])?", txt)
        if m:
            out["tg"] = (m.group(1), m.group(2) or "", m.group(3) or "")
    m = re.search(r"pp256=([\d.,]+)", txt)
    if m:
        out["pp256"] = m.group(1).replace(",", "")
    m = re.search(r"pp4096=([\d.,]+)", txt)
    if m:
        out["pp4096"] = m.group(1).replace(",", "")
    return out


def measurements():
    rows = []
    for r in load("phaseA.jsonl"):
        n = r["name"]
        tier = r.get("tier")
        d = parse_perf_out(r.get("out"))
        tg = d.get("tg", ("", "", ""))
        rows.append({
            "phase": "A_offline_perf.py",
            "tier": num(tier),
            "arm": n,
            "n_runs": 1,
            "tg_tps_median": tg[0], "tg_tps_min": tg[1], "tg_tps_max": tg[2],
            "pp256": d.get("pp256", ""), "pp4096": d.get("pp4096", ""),
            "pp_tps_median": "", "pp_tps_min": "", "pp_tps_max": "",
            "boot_s": "", "boot_s_min": "", "boot_s_max": "",
            "vram_peak_mb": num(r.get("vramPeak")),
            "vram_minfree_mb": num(r.get("vramMinFree")),
            "ram_free_after_mb": "",
            "status": r.get("status"),
            "source": "vram/status: phaseA.jsonl | throughput: perf.py .out",
            "note": "tg0/pp* come from the ONE surviving .out for this arm "
                    "(the harness writes <arm>.out, so each tier re-run "
                    "overwrote the previous). The tier column describes the "
                    "JSONL ROW, not the run that produced the .out -- do not "
                    "read tg0 as tier-specific.",
        })
    for r in load("phaseB.jsonl"):
        rows.append({
            "phase": "B_offline_spec_decode.py",
            "tier": num(r.get("tier")),
            "arm": r["name"],
            "n_runs": 1,
            "tg_tps_median": "", "tg_tps_min": "", "tg_tps_max": "",
            "pp256": "", "pp4096": "",
            "pp_tps_median": "", "pp_tps_min": "", "pp_tps_max": "",
            "boot_s": "", "boot_s_min": "", "boot_s_max": "",
            "vram_peak_mb": num(r.get("vramPeak")),
            "vram_minfree_mb": num(r.get("vramMinFree")),
            "ram_free_after_mb": "",
            "status": r.get("status"),
            "throughput_tier_known": "no",
            "source": "phaseB.jsonl",
            "note": "tg/acc per arm are in draft.csv (sweep run, "
                    "baseline-bracketed).",
        })
    for r in load("phaseC.jsonl"):
        tg, pp = r.get("sustTgMedian"), r.get("sustPpMedian")
        if tg is None:
            continue
        rows.append({
            "phase": "C_live_server",
            "tier": "",
            "arm": r["arm"],
            "n_runs": 1,
            "tg_tps_median": num(tg), "tg_tps_min": "", "tg_tps_max": "",
            "pp256": "", "pp4096": "",
            "pp_tps_median": num(pp), "pp_tps_min": "", "pp_tps_max": "",
            "boot_s": num(r.get("bootToFirstTokenSec")),
            "boot_s_min": "", "boot_s_max": "",
            "vram_peak_mb": num(r.get("vramPeak")),
            "vram_minfree_mb": num(r.get("vramMinFreeDuringLoad")),
            "ram_free_after_mb": num(r.get("ramFreeAfter")),
            "status": r.get("exit") or "OK",
            "source": "phaseC.jsonl",
            "throughput_tier_known": "yes",
            "note": "arms are runs, not arm aggregates -- aggregate by arm",
        })
    return rows


MEAS_FIELDS = ["phase", "tier", "arm", "n_runs",
               "tg_tps_median", "tg_tps_min", "tg_tps_max",
               "pp256", "pp4096",
               "pp_tps_median", "pp_tps_min", "pp_tps_max",
               "boot_s", "boot_s_min", "boot_s_max",
               "vram_peak_mb", "vram_minfree_mb", "ram_free_after_mb",
               "status", "source", "throughput_tier_known", "note"]


# ------------------------------------------------------------ long context
# One row per stage/arm/boot. wall seconds are the turn-matched per-turn values
# for a 4-prompt rotate ladder (or the 5-turn conversation where noted).
LONG_CTX = [
    # stage, ptok, arm, boot, turns(s,...)            tg,   pp,   ratio, state
    ("11-16k_turnmatch", "", "baseline", 1, [3.89, 2.44, 2.27, 10.88, 9.06], None, None, 1.000, ""),
    ("11-16k_turnmatch", "", "combo", 1, [3.64, 1.94, 1.67, 7.09, 6.61], None, None, 1.360, ""),
    ("55-62k", 55239, "baseline", 1, [30.71], None, None, 1.000, ""),
    ("55-62k", 55239, "combo", 1, [30.63], None, None, 1.000, ""),
    ("55-62k", 60634, "baseline", 1, [32.85], None, None, 1.000, ""),
    ("55-62k", 60634, "combo", 1, [33.37], None, None, 0.980, ""),
    ("130k", 129998, "baseline", 1, [86.61, 86.85, 96.48, 90.65], None, None, 1.000, ""),
    ("130k", 129998, "combo", 1, [74.78, 75.44, 82.28, 79.53], 61.0, 1741, 0.860, "normal"),
    ("130k", 129910, "baseline", 1, [86.86, 86.85, 96.09, 90.54], None, None, 1.000, ""),
    ("130k", 129910, "combo", 2, [74.78, 75.44, 82.28, 79.53], None, None, 0.869, "normal"),
    ("130k", 144084, "baseline", 1, [96.28, 96.09, 96.09, 90.54], None, None, 1.000, ""),
    ("130k", 144084, "combo", 1, [82.28, 79.53], None, None, 0.855, "normal"),
    ("130k", 142617, "baseline", 1, [90.59, 90.54], None, None, 1.000, ""),
    ("130k", 142617, "combo", 1, [79.53, 82.28], None, None, 0.878, "normal"),
    ("250k", 249730, "baseline", 1, [163.21, 162.08, 162.47, 161.74], None, None, 1.000, "normal"),
    ("250k", 249730, "combo", 1, [139.97, 142.96], None, None, 0.858, "fast"),
    ("250k", 249642, "baseline", 1, [162.77, 162.28, 162.25, 162.19], None, None, 1.000, "normal"),
    ("250k", 249642, "combo", 1, [139.63, 143.45], None, None, 0.858, "fast"),
    ("250k", 251197, "baseline", 1, [168.17, 167.66, 167.11, 167.14], None, None, 1.000, "normal"),
    ("250k", 251197, "combo", 1, [149.77, 150.45], None, None, 0.890, "fast"),
    ("250k", 259145, "baseline", 1, [172.68, 172.55, 172.28, 172.71], None, None, 1.000, "normal"),
    ("250k", 259145, "combo", 1, [155.27, 156.49], None, None, 0.899, "fast"),
    ("250k", 249730, "combo", 2, [159.31, 159.74], None, None, 0.982, "slow"),
    ("250k", 249642, "combo", 2, [159.12, 159.89], None, None, 0.987, "slow"),
    ("250k", 251197, "combo", 2, [164.24, 164.88], None, None, 0.983, "slow"),
    ("250k", 259145, "combo", 2, [169.77, 170.01], None, None, 0.983, "slow"),
]


def long_context():
    rows = []
    for stage, ptok, arm, boot, turns, tg, pp, ratio, state in LONG_CTX:
        rows.append({
            "stage": stage,
            "prompt_tokens_actual": ptok,
            "arm": arm,
            "boot": boot,
            "n_turns": len(turns),
            "turn_wall_s": " ".join(f"{t:.2f}" for t in turns),
            "turn_median_s": f"{st.median(turns):.2f}",
            "tg_tps_median": num(tg),
            "pp_tps_median": num(pp),
            "ratio_vs_baseline": f"{ratio:.3f}",
            "machine_state": state,
            "confidence": ("HIGH" if stage in ("130k", "11-16k_turnmatch")
                           else "MED" if stage == "250k" else "LOW"),
            "note": ("state-matched; both arms enter fast/slow machine state"
                     if stage == "250k" else
                     "single interleaved pair" if stage == "55-62k" else ""),
        })
    return rows


LC_FIELDS = ["stage", "prompt_tokens_actual", "arm", "boot", "n_turns",
             "turn_wall_s", "turn_median_s", "tg_tps_median", "pp_tps_median",
             "ratio_vs_baseline", "machine_state", "confidence", "note"]


# ------------------------------------------------------------------- draft
DRAFT_NDT = [
    # setting, tps, vs_base_pct, acc, acc_per_draft
    ("off", 27.21, -15.3, None, None),
    ("ndt3_dyn", 35.34, 7.2, 2.30, 2.96),
    ("ndt5_dyn_prod", 33.85, 2.7, 3.00, 4.38),
    ("ndt5_dyn_prod", 32.08, -2.7, 2.76, 4.23),
    ("ndt6_dyn", 33.13, 0.5, 3.11, 4.77),
    ("ndt7_dyn", 32.68, -0.9, 3.13, 4.89),
    ("ndt8_dyn", 30.81, -6.5, 3.17, 5.29),
    ("ndt10_dyn", 31.06, -5.8, 3.31, 5.57),
]
DRAFT_ACC = [
    # category, tools, ptok_range, arm, acc_pct, acc, drafted
    ("agentic_curl", 29, "~11k", "Q4_baseline", 75.4, 1150, 1525),
    ("agentic_code", 11, "11-16k", "Q4_baseline", 68.7, 1495, 2177),
    ("agentic_code", 11, "11-16k", "FP16", 70.3, 1561, 2219),
    ("prose_translate", 0, "0.1-9k", "Q4_baseline", 61.0, 2061, 3378),
    ("agentic_curl", 29, "~11k", "2,2", 57.3, 461, 805),
    ("prose_translate", 0, "0.1-9k", "2,2", 47.2, 811, 1719),
]
DRAFT_DCM = [
    # mode, bits, ratio_vs_Q4, n, vram_minfree_mb
    ("FP16", 16, 1.021, 2, 2619),
    ("Q8", 8, 1.001, 1, 2825),
    ("Q4", 4, 1.000, 2, 3003),
    ("3,3", 3, 1.047, 1, 3067),
    ("2,2", 2, 1.30, 1, 3657),
]


def draft():
    rows = []
    for setting, tps, vs, acc, apd in DRAFT_NDT:
        rows.append({
            "test": "ndt_ladder", "setting": setting, "category": "code",
            "tools": 11, "metric": "tg_tps", "value": tps,
            "vs_baseline_pct": vs, "acceptance": num(acc),
            "acceptance_per_drafted": num(apd),
            "n": 2, "confidence": "MED",
            "note": "baseline bracketed 32.08/33.85 (5.5% spread)",
        })
    for cat, tools, ptok, arm, pct, a, d in DRAFT_ACC:
        rows.append({
            "test": "acceptance_by_category", "setting": arm,
            "category": cat, "tools": tools, "metric": "acceptance_pct",
            "value": pct, "vs_baseline_pct": "", "acceptance": a,
            "acceptance_per_drafted": d, "n": 1, "confidence": "MED",
            "note": f"ptok {ptok}; gen>=200 only; content/tool-count CONFOUNDED",
        })
    for mode, bits, ratio, n, vmin in DRAFT_DCM:
        rows.append({
            "test": "draft_cache_mode_ladder", "setting": mode,
            "category": "code", "tools": 11, "metric": "turn_matched_ratio",
            "value": ratio, "vs_baseline_pct": "", "acceptance": "",
            "acceptance_per_drafted": "", "n": n, "confidence": "MED",
            "note": f"bits={bits}; vram_minfree={vmin}MB; baseline Q4=1.000",
        })
    return rows


DRAFT_FIELDS = ["test", "setting", "category", "tools", "metric", "value",
                "vs_baseline_pct", "acceptance", "acceptance_per_drafted",
                "n", "confidence", "note"]


# --------------------------------------------------------------- decisions
DECISIONS = [
    ("EXL3_MOE_CPU_THREADS", "start_tuned.ps1", "8", "16", "ADOPT",
     "+26.1% offline decode; +35.4% live tg; 1.026x at 130k (prefill-neutral)",
     "none (no VRAM)", "HIGH",
     "Largest single win. KEEP if VRAM binds; costs nothing."),
    ("cpu_moe_split_experts", "config.yml", "(absent, mcl38)", "380", "ADOPT",
     "1.135x alone at 130k (prefill 1507->1725 T/s, +14%); 1.157x combined",
     "~2.5 GB VRAM (min-free 1479-1543MB at 96-99% cache use)", "HIGH at 130k / MED short-ctx",
     "Floor is measured: 375 boots but is 0.3% (noise) for 35% less margin; 360 will not boot."),
    ("draft_num_tokens", "config.yml", "5", "3", "ADOPT",
     "+7.2% tg, baseline-bracketed sweep 3-10", "none", "HIGH",
     "Quality is unaffected by drafting, so acceptance is a speed-only constraint. Ceiling above 5 closed: speed decays past ndt6."),
    ("cache_mode", "config.yml", "5,4", "2,2", "ADOPT",
     "-2.1% decode (noise), +5.2% pp4096, +998 MB VRAM", "none", "HIGH (null result)",
     "Kept as VRAM relief, not as a speed change."),
    ("draft_cache_mode", "config.yml", "Q4", "Q4", "KEEP",
     "Q8 1.001 and FP16 1.021 both neutral but cost 178/384 MB; 2,2 and 3,3 slower in all 3 categories",
     "none", "MED",
     "Operator chose Q4 deliberately to recover VRAM without hurting tg. Ladder confirms the choice."),
    ("dynamic_draft", "config.yml", "true", "true", "KEEP",
     "static is -7.1% tg despite HIGHER acceptance (2.92/5.00 vs 2.78/4.29)", "none", "MED", ""),
    ("cuda_malloc_async", "config.yml", "True", "True", "KEEP", "False costs ~760 MB VRAM, no gain", "none", "MED", "Differs from upstream default; do not 'fix'."),
    ("cpu_moe_offload_layers", "config.yml", "38", "(removed)", "REPLACE",
     "mcl34 is faster (Tier 0 +22.3%) but UNSAFE: 129 MB free, cudaErrorLaunchFailure. mcl32 will not load.",
     "n/a", "HIGH", "Mutually exclusive with cpu_moe_split_experts. Never set both."),
    ("cpu_moe_offload_layers", "config.yml", "36", "(keep 38)", "REJECT",
     "-1.5% live tg, +1.7 GB VRAM", "+1.7 GB", "MED", "Direction is down; 36/38/40/42 all tried."),
    ("mcs value below 380", "config.yml", "-", "375", "REJECT",
     "boots (967 MB free) but 38.5 vs 38.4 tok/s = 0.3% noise", "35% less VRAM margin", "MED",
     "Offline predicted +2.4% decode; live delivered none."),
    ("mcs 390 / 405", "config.yml", "-", "390, 405", "REJECT",
     "SLOWER (26.68 / 23.60 tg0) AND smaller footprint (4075 / 5599 MB free)",
     "gains VRAM, loses speed", "MED", "Never live-booted; offline only."),
    ("sysmem_kv_cache", "config.yml", "0", "0", "KEEP",
     "8192 -> RAM guard 492 MB free; 24576 -> RAM guard 88 MB free", "n/a", "HIGH",
     "INFEASIBLE. Allocated eagerly from the ~34 GB expert arena. RAM is the binding constraint."),
    ("recurrent_checkpoint_interval", "config.yml", "(unset)", "(unset)", "KEEP",
     "4096 -> ratio 1.013, VRAM unchanged; 512 -> +1.7% pp for +800 MB", "none if unset", "MED",
     "Neither end earns its place."),
    ("recurrent_checkpoint_interval_pp", "config.yml", "32768", "(keep)", "KEEP",
     "8192 gives 6.7% cheaper early edits vs 0.7% on default (confirmed: 11% vs 6% cached)",
     "2.3 GiB RAM", "MED",
     "WORKS but not adopted: RAM is the binding constraint. One-line change if edit-heavy work becomes the norm."),
    ("EXL3_MOE_CPU_PIN", "start_tuned.ps1", "1", "1", "KEEP", "-13.0% to -19.8% without", "none", "MED", ""),
    ("EXL3_MOE_CPU_SWIZZLE", "start_tuned.ps1", "1", "1", "KEEP", "-3.6% to -11.2% without", "none", "MED", ""),
    ("EXL3_MOE_ZERO_COPY", "start_tuned.ps1", "1", "1", "KEEP (low conf)",
     "-2.5% to +3.5% -- straddles zero across sources", "none", "LOW", "Never measured at any tier."),
    ("EXL3_MOE_STREAM_T", "start_tuned.ps1", "6", "6", "KEEP",
     "STREAM_T=12 costs pp256 -40%; STREAM_T=3 is +1.5% tg (noise)", "none", "MED", ""),
    ("EXL3_MOE_STREAM_BATCH_EXPERTS", "start_tuned.ps1", "48", "48", "KEEP", "24 -> +2.9% tg (noise)", "none", "LOW", ""),
    ("EXL3_MOE_MEMOPS", "start_tuned.ps1", "0", "0", "SKIP", "not re-measured; plan says 29%, PERF_FINDINGS says +10%", "none", "SCREEN",
     "Sources disagree; deliberately not measured."),
    ("warmup", "config.yml", "true", "true", "KEEP", "+12 s boot, no sustained gain, REDUCES VRAM ~40-80 MB", "+12 s boot", "HIGH", "CUDA graphs are not a VRAM cost."),
    ("vision_offload", "config.yml", "true", "true", "KEEP", "false costs ~1.1 GB VRAM, no speed change", "none", "HIGH",
     "Frees only ~150-190 MB, not the ~1.1 GB its fp16 weights suggest."),
    ("max_batch_size", "config.yml", "2", "2", "KEEP", "1 -> +1.9% tg but HALVES serving capacity", "halves capacity", "LOW", "Do not take on speed alone."),
    ("chunk_size", "config.yml", "4096", "4096", "KEEP", "8192 -> +4.7 GB VRAM, no gain", "none", "LOW", ""),
]


def decisions():
    return [{
        "knob": k, "where": w, "current": c, "proposed": p, "verdict": v,
        "measured_effect": g, "cost": cost, "confidence": conf, "rationale": r,
    } for (k, w, c, p, v, g, cost, conf, r) in DECISIONS]


DEC_FIELDS = ["knob", "where", "current", "proposed", "verdict",
              "measured_effect", "cost", "confidence", "rationale"]


# ------------------------------------------------------------- retractions
RETRACTIONS = [
    (1, "Prefill is flat / never improves at any length",
     "Read the 62k point as the general case; it was one point on a non-monotonic curve.",
     "Prefill improves at 130k (+14% T/s) and 224k."),
    (2, "The combo is ~11% SLOWER at 130k",
     "Compared arms against a baseline measured in an EARLIER, faster session window (~10% drift).",
     "With the baseline interleaved: combo is 1.157x FASTER at 130k."),
    (3, "130k is inconclusive (11.2% spread exceeds the effect)",
     "Correct observation, wrong diagnosis -- blamed boot-to-boot noise. Real consecutive-boot drift is 0.07%; the gap was SESSION drift.",
     "n=4 resolved it; and the reference had to be re-measured in-window."),
    (4, "EXL3_MOE_CPU_THREADS=16 is separable and safe alone, a pure decode win at every length",
     "Inferred from short-context evidence only.",
     "It IS decode-only (prefill unchanged) but worth only 1.026x at 130k; mcs380 carries the long-context win."),
    (5, "The 224k prefill win comes from KV-cache pressure pushing CPU MoE work into the prefill path",
     "Speculative mechanism, never measured.",
     "mcs380 speeds prefill +14% at 130k where the cache is only ~50% full. No paging story needed."),
    (6, "MTP acceptance tracks intrinsic token predictability; code is the most predictable case",
     "Asserted a mechanism with no measurement, on a contested premise.",
     "Measured acceptance: curl 75.4% > code 68.7% > prose 61.0%. Code is NOT the best case."),
    (7, "The config.yml md5 invariant was vacuous (0fe01cc8 = the combo config)",
     "Misread a backup filename as an applied config.",
     "config.yml.kb-<arm> is the PRISTINE snapshot taken BEFORE that arm ran. The invariant was valid all along."),
    (8, "250k is bimodal: 1.134x or 1.016x depending on the boot",
     "Compared combo's FAST machine state against baseline's NORMAL state.",
     "Both arms enter the state. Matched within state: 1.011x slow / 0.978x fast = PARITY."),
    (9, "Pagefile reads correlate with the 250k fast/slow mode",
     "Time window misaligned: jsonl ts is written at END of run, so the window captured post-run teardown.",
     "Anchored to the real sustained window: 2 of 3 SLOW boots had near-zero reads while the FAST boot had the third-highest."),
    (10, "ndt6 beating ndt5 on both axes means the knob is still rising, sweep up",
     "Treated two axes moving together as a trend signal; acceptance is not the objective.",
     "Swept 3-10: acceptance rises monotonically but speed peaks at ndt3 (+7.2%) and decays past ndt6."),
    (11, "'mcs380 is the only viable value' / '405 is backwards'",
     "Overstated: 390/405 were measured offline but NEVER live-booted. And the direction claim was right.",
     "375 tested live: boots, 0.3% gain (noise). 380 stands as the measured floor."),
]


def retractions():
    return [{"seq": s, "claim_withdrawn": c, "why_wrong": w, "correction": k}
            for s, c, w, k in RETRACTIONS]


RET_FIELDS = ["seq", "claim_withdrawn", "why_wrong", "correction"]


# ------------------------------------------------------------- methodology
METHOD = [
    ("Interleave the reference arm with every candidate",
     "Session drift is ~10%; consecutive-boot drift is 0.07%. A carried-forward baseline is invisible to any same-batch check.",
     "Two wrong 130k verdicts ('11% slower', then 'inconclusive')."),
    ("Verify the baseline is stable across its own interleaved runs before quoting a ratio",
     "The 250k batches looked conclusive until this was done.",
     "Revealed the 250k cross-state artifact."),
    ("Compare turn-matched prompts, not medians over different prompts",
     "Prompt difficulty otherwise dominates the ratio.",
     "Used for every long-context stage."),
    ("For two-role workloads (cold prefill then replay) use the within-boot ratio t2/t1",
     "Identical requests ran 20% apart across boots.",
     "Would have handed rci_pp=8192 a spurious 1.22x."),
    ("Never quote a ratio before the last interleaved arm has finished",
     "Cheapest rule, most recently learned.",
     "250k was committed as 1.144x from boot 1 of 2; settled at ~1.06x, then parity."),
    ("Verify time alignment before believing any time-correlated claim",
     "Two false correlations came from windows that did not map to the phase discussed.",
     "Killed both the 'cache pressure' theory and the pagefile correlation."),
    ("Sweep any knob whose benefit depends on a learned predictor across content categories",
     "Content changes the mechanism, not just the constants.",
     "Would have caught that code is NOT MTP's best case."),
    ("Any candidate must live-boot before being proposed",
     "The live server sits +1.7 to +2.5 GB above eval/perf.py on the same config; safe offline margin is ~2.5 GB, not the 200 MB guard.",
     "mcs360 loads offline and will not boot the server."),
    ("Predict every prompt variant before the first request at a new length",
     "actual = 0.2383*text + 294.4*msgs + 9,110. A variant over cache_size wastes a 65 s boot and looks like a config regression.",
     "A 258k file would have been ~474k tokens and could never load."),
    ("Capture anything suspected of being boot-level per boot",
     "The server log used to be <arm>.server.log, so every boot overwrote the last.",
     "The only surviving 250k lead (CPU MoE arena reservation) was unreadable for exactly this reason."),
]


def methodology():
    return [{"rule": r, "reason": w, "error_it_prevented": e} for r, w, e in METHOD]


METH_FIELDS = ["rule", "reason", "error_it_prevented"]


def write(name, fields, rows):
    path = os.path.join(OUTDIR, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  {name:<22} {len(rows):>4} rows  {len(fields)} cols")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"writing CSVs to {os.path.abspath(OUTDIR)}")
    write("measurements.csv", MEAS_FIELDS, measurements())
    write("long-context.csv", LC_FIELDS, long_context())
    write("draft.csv", DRAFT_FIELDS, draft())
    write("decisions.csv", DEC_FIELDS, decisions())
    write("retractions.csv", RET_FIELDS, retractions())
    write("methodology.csv", METH_FIELDS, methodology())
    if MISSING:
        print(f"\n  WARNING: {len(MISSING)} referenced .out file(s) not found -- "
              f"those rows ship with empty throughput:")
        for m in sorted(MISSING)[:5]:
            print(f"    {m}")
        print("    (this is a bug, not expected; check winpath())")


if __name__ == "__main__":
    main()