#!/usr/bin/env python3
"""One table of every spec_decode corpus arm, with the cache mode and split read from its argv.

The campaign's numbers live in three files (`passes.csv`, `arms.csv`, `spec_decode.csv`) and the two
things that make an arm comparable - its cache mode and its split - are buried in the `args` cell.
This script pulls them out and prints one markdown table, so the options report is generated from the
record instead of copied by hand.

Usage:
  bench/options_table.py --dir /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/results
  bench/options_table.py --dir ... --tag-prefix t14- --tag-prefix t15- --tag-prefix t16-
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys


def cache_mode(args: str) -> str:
    m = re.search(r"-cq\s+(\S+)", args)
    if not m:
        return "FP16"
    v = m.group(1)
    return v.replace(";", ",")


def flag(args: str, name: str) -> str | None:
    m = re.search(re.escape(name) + r"\s+(\S+)", args)
    return m.group(1) if m else None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required = True, help = "results directory holding the CSVs")
    ap.add_argument("--tag-prefix", action = "append", default = [], help = "only these tag prefixes")
    ap.add_argument("--runner", default = "spec_decode")
    args = ap.parse_args(argv)

    def rows(name: str) -> list[dict]:
        path = os.path.join(args.dir, name)
        if not os.path.exists(path):
            return []
        with open(path, encoding = "utf-8", errors = "replace") as f:
            return list(csv.DictReader(f))

    passes = [r for r in rows("passes.csv") if r.get("runner") == args.runner]
    if args.tag_prefix:
        passes = [r for r in passes if any(r["tag"].startswith(p) for p in args.tag_prefix)]
    arms = rows("arms.csv")
    # Latency and acceptance per run come from the ARM lines, which the harness prints once per arm.
    by_run: dict[str, dict[str, tuple]] = {}
    for a in arms:
        by_run.setdefault(a["run_id"], {})[a["arm"]] = (float(a["tok_s"]), float(a["acc"]), float(a["draft"]))

    print("| run | cache | split | corpus t/s | baseline t/s | accepted | min free VRAM | status |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(passes, key = lambda x: x["finished_utc"]):
        got = by_run.get(r["run_id"], {})
        draft = next((v for k, v in got.items() if k.lower() != "baseline"), (None, None, None))
        base = next((v for k, v in got.items() if k.lower() == "baseline"), (None, None, None))
        cache = cache_mode(r["args"])
        split = r.get("split") or flag(r["args"], "-mcs") or ""
        tps = f"{draft[0]:.2f}" if draft[0] else ""
        bps = f"{base[0]:.2f}" if base[0] else ""
        acc = f"{draft[1]:.2f}/{draft[2]:.2f}" if draft[1] else ""
        print(f"| `{r['run_id']}` | {cache} | {split} | {tps} | {bps} | {acc} | "
              f"{r['min_free_vram_mb']} MB | {r['status']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
