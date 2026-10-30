#!/usr/bin/env python3
"""Score a longctx transcript for *retrieval*, not for prose quality.

`eval/longctx.py` answers in prose and prints the reference next to it, so no script can grade the
answer. What a script can do is check whether the **facts that exist only inside the long context**
came back: a name, a plot mechanism, a passage the model was asked to find. A hit proves the model
retrieved the fact; it says nothing about style, and a miss is a real retrieval failure only if the
answer was not truncated first — so the output length is printed beside every row.

Usage:
  bench/longctx_facts.py results/longctx.csv logs/perf-<tag>-<stamp>.log
  bench/longctx_facts.py --all          # every run in longctx.csv, transcripts found by run id
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

RESULTS = "/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/results"
LOGS = "/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/logs"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Each fact is (label, all-of-groups); a group is satisfied by any of its alternatives.
# The terms are chosen to occur in the story and NOT in the instruction, so a hit means retrieval.
FACTS = {
    # a summary is allowed to paraphrase the names away; the plot is what proves it read the story
    "ic_sum":     [("Holmes/Sherlock", ("holmes", "sherlock")),
                   ("the plot (exposure / vitriol / the marriage)",
                    ("vitriol", "disfigur", "expos", "marriage", "wedding", "villain"))],
    "ic_french":  [("names it translated/French", ("transl", "french", "français", "francais")),
                   ("quotes French text", ("le ", "la ", "une ", "était", "etait", "c'était"))],
    "ic_zoomer":  [("finds the rewritten passage", ("zoomer", "slang", "rewritten", "informal"))],
    "vm_sum":     [("the time-displaced man", ("time-displaced", "from the past", "past", "anachron")),
                   ("the weapon becomes an FTL drive",
                    ("ftl", "faster-than-light", "drive", "decelerat"))],
    "vm_q1":      [("Milsom", ("milsom",)), ("unpredictable/variable", ("unpredict", "variable"))],
    "vm_q2":      [("turret", ("turret",)),
                   ("decelerate/FTL rewrite", ("decelerat", "faster-than-light", "ftl", "drive"))],
    "vm_q3":      [("interstellar/colonisation", ("interstellar", "coloniz", "colonis", "faster-than-light", "ftl"))],
    "vm_pony":    [("finds the inserted passage", ("pony", "mlp", "my little pony", "fan fiction", "inserted"))],
    "vm_char":    [(">=3 of 16 names", ("paddington", "sherikov", "gibson", "wheeler", "fredman",
                                        "goodwin", "blackwell", "hurst", "preston", "milsom",
                                        "tucker", "dickerson", "dixon", "west", "jeremy"))],
    "pp_mod":     [("finds the inserted passage", ("inserted", "unrelated", "passage"))],
    "pp_chap":    [("chapter titles", ("chapter",))],
}
MIN_NAMES = {"vm_char": 3}


def reference_strings():
    """The references the script prints beside every answer.

    They contain the very facts being looked for (the Q&A reference for `vm_q2` says "control
    turret", the summary references name Milsom and Gruner), so they are removed before scoring -
    otherwise every keyword check would hit on the prompt and prove nothing about retrieval.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    texts = os.path.join(os.path.dirname(here), "eval", "eval_texts")
    refs = []
    for name in ("illustrious_client_sum.txt", "variable_man_sum.txt", "variable_man_char.txt"):
        path = os.path.join(texts, name)
        if os.path.exists(path):
            refs.append(open(path, encoding="utf-8", errors="replace").read().strip())
    src = open(os.path.join(os.path.dirname(here), "eval", "longctx.py"),
               encoding="utf-8", errors="replace").read()
    for var in ("vm_a1", "vm_a2", "vm_a3"):
        m = re.search(var + r'\s*=\s*"([^"]+)"', src)
        if m:
            refs.append(m.group(1).strip())
    # the transcript prints the references with typographic quotes and hyphens intact; also try the
    # ASCII-folded form so a copy that normalised them still gets removed
    out = []
    for r in refs:
        out.append(r)
        out.append(r.replace("\u2019", "'").replace("\u2014", "-").replace("\u2010", "-"))
    return out


def strip(path, refs=None):
    text = open(path, encoding="utf-8", errors="replace").read().split("=== pass")[0]
    text = ANSI.sub("", text)
    for r in refs or []:
        if r and r in text:
            text = text.replace(r, "\n[reference removed]\n")
    return text


QA_SPLIT = [
    ("vm_q1", "Why do the SRB computers"),
    ("vm_q2", "What does Edward Milsom secretly do"),
    ("vm_q3", "How does humanity ultimately benefit"),
]


def sections(text):
    """Split the transcript at the test headers, and the Q&A block per question.

    The three Q&A answers share one header, so scoring them from the block would let one question's
    answer satisfy another's fact check - each is cut at its own question line instead.
    """
    heads = {
        "ic_sum": "SUMMARY TEST", "vm_sum": "SUMMARY TEST",
        "ic_french": "FRENCH TEST", "ic_zoomer": "ZOOMER TEST",
        "vm_pony": "CORRUPTION TEST", "vm_char": "NAME EXTRACTION TEST",
        "pp_mod": "EXTRA LONG TEST", "pp_chap": "EXTRA LONG TEST",
        "qa": "Q&A TEST",
    }
    order = ["ic_sum", "ic_french", "ic_zoomer", "vm_sum", "qa", "vm_pony", "vm_char",
             "pp_mod", "pp_chap"]
    raw, cursor = {}, 0
    for i, name in enumerate(order):
        start = text.find(heads[name], cursor)
        if start < 0:
            continue
        end = len(text)
        for nxt in order[i + 1:]:
            n = text.find(heads[nxt], start + len(heads[name]))
            if n > 0:
                end = n
                break
        raw[name] = text[start + len(heads[name]):end]
        cursor = start + len(heads[name])

    out = {k: v for k, v in raw.items() if k != "qa"}
    block = raw.get("qa")
    if block:
        starts = []
        for job, question in QA_SPLIT:
            pos = block.find(question)
            starts.append((pos, job))
        starts.sort()
        for idx, (pos, job) in enumerate(starts):
            if pos < 0:
                continue
            end = starts[idx + 1][0] if idx + 1 < len(starts) and starts[idx + 1][0] > pos else len(block)
            out[job] = block[pos:end]
    return out


def score(name, chunk):
    facts = FACTS.get(name, [])
    low = chunk.lower()
    hits, need = 0, 0
    detail = []
    for label, alternatives in facts:
        need += 1
        if name in MIN_NAMES:
            got = sum(1 for a in alternatives if a in low)
            ok = got >= MIN_NAMES[name]
            detail.append(f"{label}:{'ok' if ok else 'miss'}({got})")
        else:
            ok = any(a in low for a in alternatives)
            detail.append(f"{label}:{'ok' if ok else 'miss'}")
        hits += bool(ok)
    return hits, need, " ".join(detail)


def main(argv):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--all", action="store_true", help="score every run in longctx.csv")
    ap.add_argument("csv_path", nargs="?", default=os.path.join(RESULTS, "longctx.csv"))
    ap.add_argument("log_path", nargs="?")
    args = ap.parse_args(argv)

    rows = list(csv.DictReader(open(args.csv_path, newline="", encoding="ascii", errors="replace")))
    if args.all:
        by_run = {}
        for r in rows:
            by_run.setdefault(r["run_id"], []).append(r)
        targets = sorted(by_run)
    else:
        if not args.log_path:
            ap.error("give a transcript, or --all")
        targets = [os.path.basename(args.log_path).removeprefix("perf-").removesuffix(".log")]
        by_run = {targets[0]: rows}

    for run in targets:
        log = os.path.join(LOGS, f"perf-{run}.log")
        if not os.path.exists(log):
            log = args.log_path or ""
        if not os.path.exists(log):
            print(f"{run}: no transcript at {log}")
            continue
        chunks = sections(strip(log, reference_strings()))
        print(f"\n### {run}")
        print(f"{'test':10s} {'facts':>5s} {'out_chars':>9s}  detail")
        total_hit = total_need = 0
        lengths = {r["job"]: r["out_chars"] for r in by_run.get(run, [])}
        for name in [n for n in ("ic_sum", "ic_french", "ic_zoomer", "vm_sum", "vm_q1", "vm_q2",
                                 "vm_q3", "vm_pony", "vm_char", "pp_mod", "pp_chap") if n in chunks]:
            hits, need, detail = score(name, chunks[name])
            total_hit += hits
            total_need += need
            print(f"{name:10s} {hits}/{need}      {lengths.get(name, '?'):>9s}  {detail}")
        if total_need:
            print(f"{'TOTAL':10s} {total_hit}/{total_need}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
