# _kb_report.py -- turn the knob-battery jsonl rows into the plan's report table.
#
# Reads logs\kb\phase{A,B,C}.jsonl and prints markdown. Deltas are computed only
# against a baseline row FROM THE SAME PHASE AND SAME TIER, because the plan
# (§0.5) makes tiers use different workloads -- comparing across tiers is
# meaningless. Rows without a same-tier baseline print NA rather than a number.
#
# Guard columns travel with every row: vramMinFree is what the VRAM guard
# watches (kill under 200 MB free), ramAfter is the post-run free RAM.
#
# Usage:
#   python _kb_report.py [--logdir <dir>] [--tier N] [--phase a|b|c|all]

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

BASELINE = {"a": "a00-baseline", "b": "b00-baseline", "c": "c00-baseline"}


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  !! unparsable line skipped: {line[:80]}", file=sys.stderr)
    return rows


def first_of(row, prefix):
    """First property whose name starts with prefix -> (key, value)."""
    for k, v in row.items():
        if k.startswith(prefix) and isinstance(v, (int, float)):
            return k, v
    return None, None


def win_to_mnt(p):
    """C:\\Users\\... -> /mnt/c/Users/... so the report can read box logs from WSL."""
    if not p or ":" not in p:
        return p
    drive, rest = p.split(":", 1)
    return f"/mnt/{drive.lower()}/{rest.replace(chr(92), '/')}"


ANSI = re.compile(r"\x1b\[[0-9;]*m")
PREFILL_RE = re.compile(r"^\s*Length\s+(\d+):\s+([\d.]+)\s+tokens/s")
SEQLEN_RE = re.compile(
    r"S=(\d+)\s+([\d.]+)\s+(?:tokens|it)/s(?:\s*\[([\d.]+)\s*-\s*([\d.]+)\])?")
CONTEXT_RE = re.compile(r"^\s*Context\s+(\d+):")


def parse_perf_out(path):
    """Mirror of ConvertFrom-KBPerf in _kb_lib.ps1, so a row whose parsed
    metrics were lost to an interrupted write can still be recovered from its
    stdout file. Unit flips to it/s under -sd, and one Context line carries
    every seqlen."""
    if not path or not os.path.exists(path):
        return None
    pre, gen = {}, {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = ANSI.sub("", line)
            m = PREFILL_RE.match(line)
            if m:
                pre[int(m.group(1))] = float(m.group(2))
                continue
            m = CONTEXT_RE.match(line)
            if m:
                ctx = int(m.group(1))
                gen.setdefault(ctx, {})
                for mm in SEQLEN_RE.finditer(line):
                    e = {"Tps": float(mm.group(2)), "Min": None, "Max": None}
                    if mm.group(3):
                        e["Min"] = float(mm.group(3))
                        e["Max"] = float(mm.group(4))
                    gen[ctx][int(mm.group(1))] = e
    return {"Prefill": pre, "Gen": gen}


BACKFILL = False


def backfill(row):
    """Recover tg0/pp256/pp4096 from the run's stdout when the jsonl row lacks
    them.

    OFF by default, and that is deliberate. Log file names used to collide
    across tiers (a07 at tier 1 overwrote a07 at tier 0), so backfilling a
    legacy tier-0 row from its .out silently substituted TIER 1 numbers into a
    TIER 0 table. Invoke-KBRun now writes metrics straight into the row, so
    backfill is only a recovery path for rows written before that fix -- pass
    --backfill to enable it and accept that legacy rows may be cross-polluted.
    Rows filled this way are marked so they cannot pass as first-class data."""
    if not BACKFILL or row.get("tg0") is not None:
        return row
    parsed = parse_perf_out(win_to_mnt(row.get("out", "")))
    if not parsed:
        return row
    row["_backfilled"] = True
    if 256 in parsed["Prefill"]:
        row["pp256"] = parsed["Prefill"][256]
    if 4096 in parsed["Prefill"]:
        row["pp4096"] = parsed["Prefill"][4096]
    g0 = parsed["Gen"].get(0, {}).get(1)
    if g0:
        row["tg0"] = g0["Tps"]
        if g0["Min"] is not None:
            row["tg0min"], row["tg0max"] = g0["Min"], g0["Max"]
    return row


def merge_fill(rows, extra):
    """Fill metrics missing from `rows` using `extra` (console-recovered).
    Never overrides a value the jsonl already carries: the jsonl row is the
    run's own record, the recovered row is a reconstruction. Keyed on
    (tier, name) because the same arm name exists at several tiers."""
    bykey = {(infer_tier(r), r["name"]): r for r in rows}
    filled = 0
    for e in extra:
        k = (infer_tier(e), e["name"])
        r = bykey.get(k)
        if r is None:
            rows.append(e)
            bykey[k] = e
            r = e
            filled += 1
            continue
        for fld in ("tg0", "tg0min", "tg0max", "pp256", "pp4096",
                    "status", "vramPeak", "vramMinFree", "ramAfter", "knob"):
            if r.get(fld) is None and e.get(fld) is not None:
                r[fld] = e[fld]
                r["_filled"] = True
                filled += 1
    return rows, filled


def knob_of(row):
    """Arm name when the row predates the knob field (it is self-describing:
    a01-mcl36, a04-cq22, b01-ndt4-dyn)."""
    return row.get("knob") or row["name"]


def pct(val, base):
    if val is None or base in (None, 0):
        return "NA"
    return f"{100 * (val / base - 1):+.1f}%"


def spread(lo, hi, val):
    if lo is None or hi is None or val is None:
        return f"{val:.2f}" if val is not None else "-"
    return f"{val:.2f} [{lo:.2f}-{hi:.2f}]"


def dedupe(rows):
    """Keep the LAST row per arm: a re-run supersedes an earlier attempt."""
    out = OrderedDict()
    for r in rows:
        out[r["name"]] = r
    return out


def infer_tier(row):
    """Recover the tier from the arm's own recorded argv when the `tier` field
    is absent (batches launched before that field existed). The tier is fully
    determined by -max_length, which Invoke-KBRun stores verbatim in `args`:
    1024 -> tier 0, 4096 -> tier 1, 32768 -> tier 2. Inferring beats defaulting
    to 0, which silently files tier-1 rows under tier 0 and voids the deltas."""
    if row.get("tier") is not None:
        return int(row["tier"])
    args = row.get("args") or ""
    m = re.search(r"-max_length\s+(\d+)", args)
    if m:
        return {"1024": 0, "4096": 1, "32768": 2}.get(m.group(1), 0)
    return 0


def phase_a(rows, tier):
    # Filter by tier BEFORE deduping. dedupe() keeps the last row per arm, so
    # doing it first lets a tier-1 baseline attempt overwrite the tier-0
    # baseline and silently voids every tier-0 delta.
    rows = [r for r in rows if infer_tier(r) == tier]
    rows = [backfill(r) for r in dedupe(rows).values()]
    if not rows:
        print(f"### Phase A -- no rows at tier {tier}\n")
        return
    base = next((r for r in rows
                 if r["name"] == BASELINE["a"] and r["status"] == "OK"), None)
    if base is None:
        print(f">> No successful baseline at tier {tier}; deltas omitted.\n")
    btg = base.get("tg0") if base else None
    bpp = base.get("pp4096") if base else None
    print(f"### Phase A -- engine knobs, eval/perf.py, server stopped (tier {tier})\n")
    print("Baseline = production config via CLI: "
          "`-mcl 38 -cq 5,4 -cs 262144 -chunk_size 4096 -ambs 2`\n")
    print("| arm | knob | status | tg0 tok/s [min-max] | tg0 vs base | pp256 | pp4096 | pp4096 vs base | VRAM peak | VRAM min free | RAM after |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        tg = r.get("tg0")
        lo, hi = r.get("tg0min"), r.get("tg0max")
        pp256, pp4096 = r.get("pp256"), r.get("pp4096")
        status = r["status"]
        if status != "OK":
            status = f"**{status}** {r.get('guard') or ''}"
        bflag = (" (recovered)" if r.get("_filled") else
                 " (backfilled)" if r.get("_backfilled") else "")
        print(f"| {r['name']}{bflag} | {knob_of(r)} | {status} | "
              f"{spread(lo, hi, tg)} | {pct(tg, btg)} | "
              f"{f'{pp256:.1f}' if pp256 else '-'} | "
              f"{f'{pp4096:.1f}' if pp4096 else '-'} | {pct(pp4096, bpp)} | "
              f"{r.get('vramPeak','-')} MB | {r.get('vramMinFree','-')} MB | "
              f"{r.get('ramAfter','-')} MB |")
    print()


def phase_b(rows, tier):
    rows = [r for r in rows if infer_tier(r) == tier]
    rows = list(dedupe(rows).values())
    base = next((r for r in rows if r["name"] == BASELINE["b"] and r["status"] == "OK"), None)
    btps = first_of(base, "tps_")[1] if base else None
    print(f"### Phase B -- draft knobs, eval/spec_decode.py, server stopped (tier {tier})\n")
    print("| arm | knob | status | tg tok/s | vs base | acc/draft | VRAM peak | VRAM min free | RAM after |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        tps = first_of(r, "tps_")[1]
        acc = first_of(r, "acc_")[1]
        drf = first_of(r, "draft_")[1]
        status = r["status"]
        if status != "OK":
            status = f"**{status}** {r.get('guard') or ''}"
        accs = f"{acc:.2f}/{drf:.2f}" if acc is not None and drf is not None else "-"
        bflag = (" (recovered)" if r.get("_filled") else
                 " (backfilled)" if r.get("_backfilled") else "")
        print(f"| {r['name']}{bflag} | {knob_of(r)} | {status} | "
              f"{f'{tps:.2f}' if tps else '-'} | {pct(tps, btps)} | {accs} | "
              f"{r.get('vramPeak','-')} MB | {r.get('vramMinFree','-')} MB | "
              f"{r.get('ramAfter','-')} MB |")
    print()


def phase_c(rows):
    # NOTE: phaseC.jsonl rows carry no tgTps/ppTps -- the per-request metrics
    # live only in each arm's console log, because Add-Content runs before the
    # log is flushed and parsed. Keep this table honest rather than printing
    # empty columns; the sustained numbers come from the console extract.
    rows = dedupe(rows).values()
    print("### Phase C -- live server, boot + server-only knobs\n")
    print("| arm | config applied | boot->ready s | ready->1st tok s | **boot->1st tok** | load s | VRAM peak | VRAM min free | VRAM free after | RAM before | RAM after | reqs |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cfg = r.get("configApplied") or "none (baseline)"
        env = r.get("envOverrides") or ""
        if env:
            cfg += f" ; env {env}"
        print(f"| {r.get('arm','?')} | {cfg} | {r.get('bootReadySec','-')} | "
              f"{r.get('readyToFirstTokenSec','-')} | **{r.get('bootToFirstTokenSec','-')}** | "
              f"{r.get('loadSec','-')} | {r.get('vramPeak','-')} MB | "
              f"{r.get('vramMinFreeDuringLoad','-')} MB | {r.get('vramFreeAfter','-')} MB | "
              f"{r.get('ramFreeBefore','-')} MB | {r.get('ramFreeAfter','-')} MB | "
              f"{r.get('serverRequests','-')} |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=r"C:\Users\yoho\Downloads\tabbyAPI\logs\kb")
    ap.add_argument("--tier", type=int, default=0)
    ap.add_argument("--phase", default="all")
    ap.add_argument("--merge", action="append", default=[],
                    help="extra jsonl of console-recovered rows to fill gaps "
                         "(never overrides values the jsonl already has)")
    ap.add_argument("--backfill", action="store_true",
                    help="recover metrics from .out files for legacy rows whose "
                         "jsonl predates in-row metrics (may be cross-tier polluted)")
    args = ap.parse_args()
    global BACKFILL
    BACKFILL = args.backfill

    print(f"<!-- knob battery report -- logdir {args.logdir} tier {args.tier} -->\n")
    if args.phase in ("a", "all"):
        arows = load(os.path.join(args.logdir, "phaseA.jsonl"))
        for extra_path in args.merge:
            erows = load(extra_path)
            arows, n = merge_fill(arows, erows)
            print(f"<!-- merged {n} fields from {extra_path} -->")
        phase_a(arows, args.tier)
    if args.phase in ("b", "all"):
        phase_b(load(os.path.join(args.logdir, "phaseB.jsonl")), args.tier)
    if args.phase in ("c", "all"):
        phase_c(load(os.path.join(args.logdir, "phaseC.jsonl")))


if __name__ == "__main__":
    main()