#!/usr/bin/env python3
"""Compare a set of runs on this box's bands, with the within-run spread attached.

Reads results/pp.csv, tg.csv, passes.csv and prints, per tag: PP at the long bands, TG S=1 at the
long bands, the min free VRAM/RAM, and - when the run used `-dr N` - the min-max spread perf.py
printed for those contexts. Nothing here decides anything; it puts a verdict's evidence on one
screen so a ratio is quoted against a spread instead of against a single number.

Usage:
  bench/compare_runs.py --ref t9-ref-a --tag t9-th15-a --tag t9-graphs0-a
  bench/compare_runs.py --ref t9-ref-a --tag t9-th15-a --bands 65536 130816
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics

RESULTS = "/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/results"
LOGS = "/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/logs"


def read(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path, newline="", encoding="ascii", errors="replace")))


def spread_from_log(run_id):
    """perf.py prints `Context <n>: S=1 <mean> tokens/s [<min> - <max>]` when -dr > 1.

    The context index and the S=1 value are on the SAME line, so both are parsed together; the
    line also carries ANSI colour codes and a unit between the mean and the bracket.
    """
    path = os.path.join(LOGS, f"perf-{run_id}.log")
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        if "Context" not in line:
            continue
        clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
        ctx = re.match(r"\s*Context\s+(\d+):", clean)
        if not ctx:
            continue
        m = re.search(r"S=1\s+([0-9.]+)\s*\S*\s*\[([0-9.]+)\s*-\s*([0-9.]+)\]", clean)
        if m:
            mean, lo, hi = (float(x) for x in m.groups())
            out[ctx.group(1)] = (lo, mean, hi)
    return out


def main(argv):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--ref", required=True, help="reference tag")
    ap.add_argument("--tag", action="append", required=True, help="tag to compare (repeatable)")
    ap.add_argument("--bands", nargs="*", default=["65536", "130816"], help="TG contexts to print")
    ap.add_argument("--pp-bands", nargs="*", default=["65536", "131072"], help="PP bands to print")
    args = ap.parse_args(argv)

    pas = {r["tag"]: r for r in read("passes.csv")}
    pp, tg = {}, {}
    for r in read("pp.csv"):
        pp.setdefault(r["tag"], {})[r["band"]] = float(r["pp_tok_s"])
    for r in read("tg.csv"):
        tg.setdefault(r["tag"], {})[r["context"]] = float(r["s1"])

    tags = [args.ref] + args.tag
    print(f"{'tag':22s} {'PP ' + '/'.join(args.pp_bands):34s} {'TG ' + '/'.join(args.bands):34s} min-vram min-ram")
    for tag in tags:
        if tag not in pas:
            print(f"{tag:22s} (no such run)")
            continue
        r = pas[tag]
        p = " ".join(f"{pp.get(tag, {}).get(b, float('nan')):8.2f}" for b in args.pp_bands)
        t = " ".join(f"{tg.get(tag, {}).get(b, float('nan')):8.2f}" for b in args.bands)
        print(f"{tag:22s} {p:34s} {t:34s} {r.get('min_free_vram_mb','?'):>8s} {r.get('min_free_ram_mb','?'):>7s}")

    print()
    for band in args.bands:
        ref = tg.get(args.ref, {}).get(band)
        if not ref:
            continue
        print(f"TG {band} ratios vs {args.ref} ({ref:.2f}):", end="")
        for tag in args.tag:
            v = tg.get(tag, {}).get(band)
            if v:
                sp = spread_from_log(pas.get(tag, {}).get("run_id", ""))
                lo_hi = f" [{sp[band][0]:.1f}-{sp[band][2]:.1f}]" if band in sp else ""
                print(f"  {tag}={v/ref:.3f}{lo_hi}", end="")
        print()
    for band in args.pp_bands:
        ref = pp.get(args.ref, {}).get(band)
        if not ref:
            continue
        print(f"PP {band} ratios vs {args.ref} ({ref:.0f}):", end="")
        for tag in args.tag:
            v = pp.get(tag, {}).get(band)
            if v:
                print(f"  {tag}={v/ref:.3f}", end="")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
