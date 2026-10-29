#!/usr/bin/env python3
"""Recover spec_decode results from logs made before the harness parsed them.

`run_pass.ps1` now writes `results/spec_decode.csv` directly, but runs recorded earlier exist only
as logs. This reads those logs with the same rules the harness uses (the rendered per-category table
and the `ARM ...` lines) and appends any missing `(run_id, category, arm)` rows, so the drafting
series is analysable by `bench/ratios.py` like every other measurement.

Idempotent: rows already present for a run id are left alone.

Usage:
  bench/backfill_spec_csv.py --logs /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/logs \
                             --out   /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/results/spec_decode.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TAG_FROM_LOG = re.compile(r"^perf-(.+)\.log$", re.S)


def parse_log(path: str) -> tuple[str, list[list[str]]]:
    text = ANSI.sub("", open(path, encoding="utf-8", errors="replace").read())
    m = TAG_FROM_LOG.match(os.path.basename(path))
    run_id = m.group(1) if m else os.path.basename(path)
    rows: list[list[str]] = []
    header = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and re.match(r"^\|\s*Category\s*\|", line):
            cells = [c.strip() for c in line.strip("|").split("|")]
            header = cells[1:]
            continue
        if header is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(header) + 1 or re.match(r"^-+$", cells[0]):
            continue
        for arm, cell in zip(header, cells[1:]):
            mm = re.search(r"([0-9.]+)\s+t/s", cell)
            if not mm:
                continue
            ratio = re.search(r"([0-9.]+)x", cell)
            accd = re.search(r"([0-9.]+)/([0-9.]+)\s+acc/draft", cell)
            rows.append([
                run_id, cells[0], arm, mm.group(1),
                ratio.group(1) if ratio else "0",
                accd.group(1) if accd else "-1",
                accd.group(2) if accd else "-1",
            ])
    return run_id, rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    ap.add_argument("--logs", required=True, help="directory holding perf-<run_id>.log files")
    ap.add_argument("--out", required=True, help="spec_decode.csv to append to")
    ap.add_argument("--glob", default="perf-*.log", help="log glob (default perf-*.log)")
    args = ap.parse_args(argv)

    existing = set()
    if os.path.exists(args.out):
        with open(args.out, newline="", encoding="ascii", errors="replace") as f:
            for r in csv.DictReader(f):
                existing.add((r.get("run_id"), r.get("category"), r.get("arm")))
    else:
        with open(args.out, "w", newline="", encoding="ascii") as f:
            f.write("tag,run_id,category,arm,tok_s,ratio,acc,draft\n")

    added = 0
    for path in sorted(glob.glob(os.path.join(args.logs, args.glob))):
        if path.endswith(".err.log"):
            continue
        run_id, rows = parse_log(path)
        tag = run_id.rsplit("-", 1)[0] if "-" in run_id else run_id
        for run_id_, category, arm, tok_s, ratio, acc, draft in rows:
            if (run_id_, category, arm) in existing:
                continue
            with open(args.out, "a", newline="", encoding="ascii") as f:
                f.write(f"{tag},{run_id_},{category.replace(',', ';')},{arm.replace(',', ';')},"
                        f"{tok_s},{ratio},{acc},{draft}\n")
            existing.add((run_id_, category, arm))
            added += 1
    print(f"backfilled {added} rows into {args.out} ({len(existing)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
