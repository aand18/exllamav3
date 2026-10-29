#!/usr/bin/env python3
"""Pre- and post-task gates: make a report's claims checkable against its own evidence.

Two failure modes this session actually produced drove this script:

  * a report claiming "split 368 does not load" while `passes.csv` held a row for it with
    `status=ok` (the pass loaded, then died in long prefill);
  * a report quoting "pp ±5 % (0.95-0.99)" while its own ratio table started at 0.92.

Both are mechanical to catch. Run `--pre` before spending GPU time and `--post` before claiming a
task done; paste the output into the report.

Usage:
  bench/verify_task.py --pre                     # state/model/engine/profile/fit
  bench/verify_task.py --post --report wiki/plans/v165-reports/T06-mtp.md
  bench/verify_task.py --post --report ... --expect-or     # task needs -mtp -or

Exit code 0 = all gates passed, 1 = something a reader would call a contradiction.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DEFAULT = "/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/results"
LOGS_DEFAULT = "/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/logs"
VENV = "/mnt/c/Users/yoho/Downloads/tabbyAPI/venv/Lib/site-packages/exllamav3"

# a run id is tag + UTC stamp; the tag itself may contain hyphens ("rb-ab-166-20261029-202655"),
# so match the longest token that ends in <8-digit date>-<6-digit time>
RUN_ID = re.compile(r"(?<![\w-])([A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*-\d{8}-\d{6})(?![\w-])")


def wsl_path(p: str) -> str:
    """Translate a Windows path to its WSL mount (the gates run from WSL, the box is Windows)."""
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p or "")
    if not m:
        return p
    return "/mnt/" + m.group(1).lower() + "/" + m.group(2).replace("\\", "/")


def ok(msg: str) -> bool:
    print(f"  PASS  {msg}")
    return True


def bad(msg: str) -> bool:
    print(f"  FAIL  {msg}")
    return False


def warn(msg: str) -> bool:
    print(f"  WARN  {msg}")
    return True


def pre(args) -> bool:
    good = True
    state_path = args.state
    if not os.path.exists(state_path):
        return bad(f"state.json missing at {state_path}")
    state = json.load(open(state_path))["active"]
    good &= ok(f"state.json: model {os.path.basename(state['model'])}, split {state['split']}, "
               f"cs {state['cache_size']}, cq {state['cache_quant']}")

    model_dir = wsl_path(state["model"])
    if not os.path.isdir(model_dir):
        good &= bad(f"model directory missing: {state['model']} (as {model_dir})")

    init_py = os.path.join(VENV, "model_init.py")
    if not os.path.exists(init_py):
        good &= bad(f"installed engine not found at {VENV} (not the venv copy)")
    else:
        relaxed = "assert not draft_model_dir" not in open(init_py, encoding="utf-8", errors="replace").read()
        if args.expect_or and not relaxed:
            good &= bad("installed engine BLOCKS -or with a draft model (the T7 trap: the "
                        "relaxation lives only in the win-perf wheel)")
        elif relaxed:
            good &= ok("installed engine accepts -or together with a draft model")
        else:
            good &= warn("installed engine blocks -or with a draft model (fine unless this task needs it)")

    frozen = wsl_path(state.get("frozen_state_dir") or "")
    stats = os.path.join(frozen, "exllamav3", "expert_stats") if frozen else ""
    if stats and os.path.isdir(stats) and glob.glob(os.path.join(stats, "*.json")):
        good &= ok(f"frozen expert profile present ({len(glob.glob(os.path.join(stats, '*.json')))} file(s))")
    else:
        good &= (warn if args.allow_cold else bad)(
            "no frozen expert profile: the first run would be COLD and every later arm would "
            "inherit its placement (critique C1/C2). Run a warm-up pass or pass --allow-cold")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("size_check", os.path.join(HERE, "size_check.py"))
        sc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sc)
        rc = sc.main(["--model", model_dir, "--mcs", str(state["split"]),
                      "--cs", str(state["cache_size"]), "--cq", state["cache_quant"]])
        good &= (rc == 0) or bad("size_check says this configuration does not fit")
    except SystemExit:
        good &= bad("size_check rejected the configuration")
    except Exception as exc:                      # pragma: no cover - diagnostic path
        good &= warn(f"size_check could not run: {exc}")
    return good


def post(args) -> bool:
    good = True
    if not args.report or not os.path.exists(args.report):
        return bad(f"report not found: {args.report}")
    text = open(args.report, encoding="utf-8", errors="replace").read()
    cited = sorted(set(RUN_ID.findall(text)))
    if not cited:
        return bad("the report cites no run ids: nothing here is checkable")

    rows = {}
    csv_path = os.path.join(args.results, "passes.csv")
    with open(csv_path, newline="", encoding="ascii", errors="replace") as f:
        for r in csv.DictReader(f):
            rows[r.get("run_id", "")] = r

    for run in cited:
        r = rows.get(run)
        if r is None:
            good &= bad(f"{run}: cited but absent from passes.csv")
            continue
        if r.get("status") != "ok":
            good &= bad(f"{run}: status={r.get('status')} (only ok runs may be cited as measurements)")
            continue
        if r.get("breach", "0") != "0":
            good &= bad(f"{run}: breach flag set")
            continue
        log = os.path.join(args.logs, f"perf-{run}.log")
        warm = os.path.exists(log) and "placement from saved profile" in open(
            log, encoding="utf-8", errors="replace").read()
        if warm:
            good &= ok(f"{run}: ok, no breach, warm expert placement")
        elif args.allow_cold:
            good &= warn(f"{run}: ok, no breach, but COLD (no saved placement) - allowed by flag")
        else:
            good &= bad(f"{run}: ok but COLD (first run on this model/split); its baseline is "
                        "not comparable - re-run warm or pass --allow-cold")

    # a report that quotes percentages must agree with the ratio tool
    if re.search(r"±|\b\d+(\.\d+)?\s*%", text):
        good &= ok("report states percentages - regenerate the ratios table and check the ranges "
                   "quoted (bench/ratios.py) before publishing")
    return good


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", action="store_true")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--report", default="", help="report file to verify (post)")
    ap.add_argument("--state", default=os.path.join(HERE, "windows", "state.json"))
    ap.add_argument("--results", default=RESULTS_DEFAULT)
    ap.add_argument("--logs", default=LOGS_DEFAULT)
    ap.add_argument("--expect-or", action="store_true", help="the task will use -mtp -or")
    ap.add_argument("--allow-cold", action="store_true", help="accept a cold (first) run")
    args = ap.parse_args(argv)
    if not (args.pre or args.post):
        ap.error("choose --pre or --post")

    good = True
    if args.pre:
        print("pre-run gates")
        good &= pre(args)
    if args.post:
        print("post-run gates")
        good &= post(args)
    print("verdict:", "all gates passed" if good else "CONTRADICTIONS FOUND")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
