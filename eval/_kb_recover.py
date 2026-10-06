# _kb_recover.py -- rebuild battery rows from the CONSOLE logs.
#
# Why this exists: Invoke-KBRun writes the jsonl row BEFORE the phase script
# attaches the parsed metrics, so batches run before the write-back fix have
# jsonl rows with status but no tg0/pp. The console log is the authoritative
# record -- those numbers were printed from the same in-memory row -- so it is a
# complete recovery source. It also sidesteps the .out filename collision, where
# a later tier overwrote an earlier tier's file.
#
# Reads the per-arm console lines:
#     --- a07-chunk2048  [chunk_size]
#       -> OK 461.8s peakUsed=18998MB minFree=5141MB ramAfter=49307MB
#         tg0=24.72 [24.38-25.05] pp256=293.94 pp4096=1,231.68
#
# Usage: python _kb_recover.py /tmp/kb/phaseA_t0.log [...] [-o out.jsonl]

import argparse
import json
import re
import sys

HDR = re.compile(r"^---\s+(\S+)\s+\[(.*)\]\s*$")
STAT = re.compile(
    r"->\s+(\S+)\s+([\d.]+)s\s+peakUsed=(\d+)MB\s+minFree=(\S+)MB\s+ramAfter=(\d+)MB")
MET = re.compile(
    r"tg0=([\d.]+|NA)(?:\s+\[([\d.]+)-([\d.]+)\])?\s+"
    r"pp256=([\d.,]+|NA)\s+pp4096=([\d.,]+|NA)")


def num(s):
    if s is None:
        return None
    s = s.replace(",", "")
    if s in ("NA", "-", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse(path):
    rows, pending, tier = [], None, None

    def flush():
        # A FAIL arm never prints a tg0 line (it prints "no parse:"), so emit
        # whatever the status line gave us rather than dropping the arm.
        if pending is not None and pending.get("status"):
            rows.append(pending)

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.search(r"PHASE A tier (\d+)", line)
            if m:
                tier = int(m.group(1))
            m = HDR.match(line.rstrip())
            if m:
                flush()
                pending = {"name": m.group(1), "knob": m.group(2), "tier": tier,
                           "source": "console-recovered"}
                continue
            if pending is None:
                continue
            m = MET.search(line)
            if m:
                pending["tg0"] = num(m.group(1))
                pending["tg0min"] = num(m.group(2))
                pending["tg0max"] = num(m.group(3))
                pending["pp256"] = num(m.group(4))
                pending["pp4096"] = num(m.group(5))
                flush()
                pending = None
                continue
            m = STAT.search(line)
            if m:
                pending["status"] = m.group(1)
                pending["elapsedSec"] = float(m.group(2))
                pending["vramPeak"] = int(m.group(3))
                pending["vramMinFree"] = num(m.group(4))
                pending["ramAfter"] = int(m.group(5))
    flush()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("-o", "--out", default="-")
    args = ap.parse_args()

    merged = {}
    for p in args.logs:
        got = parse(p)
        print(f"{p}: {len(got)} rows", file=sys.stderr)
        for r in got:
            # later logs are newer runs and win
            merged[(r.get("tier"), r["name"])] = r

    rows = list(merged.values())
    out = "\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    if args.out == "-":
        print(out)
    else:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
        print(f"wrote {len(rows)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()