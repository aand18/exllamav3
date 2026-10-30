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
from decimal import Decimal, ROUND_HALF_UP

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


def info(msg: str) -> bool:
    print(f"  INFO  {msg}")
    return True


def size_check_argv(state: dict) -> list[str]:
    """The `size_check.py` argv for a regime, draft head included.

    A fit check that leaves `--draft mtp` off measures a smaller machine than the one about to be
    loaded: the head brings its own GPU share, its own pinned host arena and its own one-layer
    cache (~500 MiB VRAM and ~680 MiB of arena at 3.18/376), which is the margin a pass breaches on.
    """
    argv = ["--model", wsl_path(state["model"]), "--mcs", str(state["split"]),
            "--cs", str(state["cache_size"]), "--cq", state["cache_quant"]]
    if (state.get("draft") or {}).get("mode") == "mtp":
        argv += ["--draft", "mtp"]
    return argv


def json_diff(a, b, prefix: str = "") -> list[str]:
    """Dotted paths whose parsed values differ - key order and line endings are not values."""
    out: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in a or key not in b:
                out.append(path)
            else:
                out += json_diff(a[key], b[key], path)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += json_diff(x, y, f"{prefix}[{i}]")
    elif a != b:
        out.append(prefix or "root")
    return out


def deployed_state_gate(repo: dict, deployed: str, label: str = "--state") -> bool:
    """Catch a regime change that was made in the repo and never deployed.

    The box reads `<deployed>/state.json`; `--state` validates the repo copy, so the two agree only
    because `deploy.sh` ships the file on every run. Compare parsed JSON, not bytes or hashes:
    `core.autocrlf=true` means a plain `git checkout -- state.json` writes CRLF while the blob is
    LF, so a byte comparison fails on an innocent re-checkout and proves nothing extra.
    """
    if not deployed:
        return info("deployed state.json check disabled (--deployed '')")
    path = os.path.join(deployed, "state.json")
    if not os.path.exists(path):
        return info(f"deployed state.json absent at {path}: no deployed copy to disagree with")
    try:
        with open(path, encoding="utf-8") as f:
            have = json.load(f)
    except (OSError, ValueError) as exc:
        return bad(f"deployed state.json unreadable at {path}: {exc} - run bench/windows/deploy.sh")
    if have == repo:
        return ok(f"deployed state.json matches {label} (parsed, not bytes)")
    diff = json_diff(repo, have)
    named = ", ".join(diff[:4]) + (f" (+{len(diff) - 4} more)" if len(diff) > 4 else "")
    return bad(f"deployed state.json differs from {label} at {named or 'the whole document'} - "
               "run bench/windows/deploy.sh")


def pre(args) -> bool:
    good = True
    state_path = args.state
    if not os.path.exists(state_path):
        return bad(f"state.json missing at {state_path}")
    with open(state_path, encoding="utf-8") as f:
        doc = json.load(f)
    state = doc["active"]
    good &= ok(f"state.json: model {os.path.basename(state['model'])}, split {state['split']}, "
               f"cs {state['cache_size']}, cq {state['cache_quant']}")
    good &= deployed_state_gate(doc, args.deployed, state_path)

    model_dir = wsl_path(state["model"])
    if not os.path.isdir(model_dir):
        good &= bad(f"model directory missing: {state['model']} (as {model_dir})")

    init_py = os.path.join(VENV, "model_init.py")
    if not os.path.exists(init_py):
        good &= bad(f"installed engine not found at {VENV} (not the venv copy)")
    else:
        with open(init_py, encoding="utf-8", errors="replace") as f:
            relaxed = "assert not draft_model_dir" not in f.read()
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
        rc = sc.main(size_check_argv(state))
        good &= (rc == 0) or bad("size_check says this configuration does not fit")
    except SystemExit:
        good &= bad("size_check rejected the configuration")
    except Exception as exc:                      # pragma: no cover - diagnostic path
        good &= warn(f"size_check could not run: {exc}")
    return good


PLACED = "placement from saved profile"

# a pp/tg ratio sits near 1.0; a band whose endpoint is above this is a count ("1.14-2.01 of 4
# drafts") or a throughput range ("35.2-38.8 t/s"), not a ratio, and is not checked as one
RATIO_SCALE = 1.2
SYMMETRIC_PCT = re.compile(r"±\s*\d+(?:\.\d+)?(?:\s*[–—-]\s*\d+(?:\.\d+)?)?\s*%")
PCT = re.compile(r"\d+(?:\.\d+)?\s*%")
# one or more decimals per endpoint, not two or more: a band quoted at one decimal ("0.9–1.0") beside
# a table printed at one decimal is the same claim as "0.92–0.99" and has to be read the same way -
# the 2-decimal-only pattern matched nothing there, so the report passed with no number examined
RATIO_INTERVAL = re.compile(r"(?<![\w.])(\d+\.\d+)\s*[–—-]\s*(\d+\.\d+)(?!\w)(?!\.\d)")
# a ratio cell printed with 1 to 3 decimals; a band, a t/s rate (45.76) or a MiB figure (2141) is
# not one, which is what keeps an unrelated table from being read as a ratio table
RATIO_CELL = re.compile(r"[01]\.\d{1,3}")
# a ratio range written in one cell ("0.915–1.034"): a per-band pair of measurements rather than one
# median, so no column of that table yields a min/max band for a quote to be checked against
RATIO_RANGE_CELL = re.compile(r"[01]\.\d{1,3}\s*[–—-]\s*[01]\.\d{1,3}")


def warm_gate(log: str, label: str, allow_cold: bool = False) -> bool:
    """Every CPU-split component must report a saved placement, not one of them.

    A substring test over the whole log is what certified `perf-mtp-ref-318-20261029-222244` warm:
    49 `CPU split experts` lines, 48 of them from the saved profile, and the cold one is the MTP
    head on line 2 - the component the drafting arms are measured against. Placement is per
    component, so the check is per line.
    """
    if not os.path.exists(log):
        return bad(f"{label}: log missing at {log} - nothing to check the placement against")
    with open(log, encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if "CPU split experts" in line]
    if not lines:
        return bad(f"{label}: no `CPU split experts` line - this run reports no expert placement, "
                   "so its warmth is unproven")
    cold = [line for line in lines if PLACED not in line]
    if not cold:
        return ok(f"{label}: {len(lines)} CPU-split component(s), every one from the saved profile")
    detail = "\n".join("          " + line for line in cold)
    message = (f"{label}: {len(cold)} of {len(lines)} CPU-split component(s) have no saved "
               f"placement, so their placement was computed fresh and is not comparable:\n{detail}")
    if allow_cold:
        return warn(message)
    return bad(message)


def arm_runs(results: str) -> tuple[set[str], str]:
    """Run ids that produced at least one arm row, and the file(s) that said so.

    `arms.csv` is the machine-readable arm table; `spec_decode.csv` is the per-category table that
    predates it, and the archive's older drafting runs exist only there (their logs carry no `ARM`
    line). A run that died before its first arm is in neither.
    """
    ids: set[str] = set()
    sources: list[str] = []
    for name in ("arms.csv", "spec_decode.csv"):
        path = os.path.join(results, name)
        if not os.path.exists(path):
            continue
        sources.append(name)
        with open(path, newline="", encoding="ascii", errors="replace") as f:
            for r in csv.DictReader(f):
                ids.add((r.get("run_id") or "").strip())
    return ids, " or ".join(sources)


def arm_gate(run: str, arms: set[str], source: str) -> bool:
    """A run that produced no arm is not a measurement, whatever its exit code said."""
    if not source:
        return bad(f"{run}: runner=spec_decode but this results directory has no arms.csv and no "
                   "spec_decode.csv - there is no arm evidence to certify")
    if run in arms:
        return ok(f"{run}: arm rows present in {source}")
    return bad(f"{run}: runner=spec_decode but no row for it in {source} - the run died before "
               "producing an arm, so it is not a measurement")


def row_runner(r: dict) -> str:
    """The runner that wrote a `passes.csv` row.

    `runner` is appended after `log`, and the box's passes.csv was created before that column
    existed, so its old 20-column header leaves the value in DictReader's overflow list.
    """
    return (r.get("runner") or (r.get(None) or [""])[0] or "").strip()


def markdown_cells(line: str) -> list[str]:
    """Cells of a markdown table row, decoration stripped, or [] for any other line."""
    s = line.strip()
    if len(s) < 4 or not (s.startswith("|") and s.endswith("|")):
        return []
    return [c.strip().strip("*`").strip().removesuffix("×").strip() for c in s[1:-1].split("|")]


def ratio_columns(text: str) -> list[tuple[float, float]]:
    """(min, max) of every ratio column in the report's own ratio tables, at 2 decimals.

    A ratio table is a run of markdown rows whose last cell is a ratio (`0.92`), which is how
    `bench/ratios.py` prints them - its `fmt()` writes `-` for a missing ratio, so a table can hold
    rows that are not ratio rows. Only a non-table line ends a table: the range is the min/max over
    that table's ratio rows, never over the fragment between two gaps, or a quote of the survivors
    after a gap would be certified against the wrong range.

    Every column that is a ratio in every one of those rows is a candidate, not just the last: a
    report that prints two ratio columns (pp and tg) quotes a band from either. Recognised ratio
    cells are `0.`- or `1.`-prefixed decimals of 1 to 3 places (`RATIO_CELL`); a t/s rate, a MiB
    figure or a band number is not a ratio cell.
    """
    tables: list[list[list[str]]] = []
    table: list[list[str]] = []
    for line in text.splitlines():
        cells = markdown_cells(line)
        if cells:
            table.append(cells)                        # separator and header rows included
        elif table:
            tables.append(table)
            table = []
    if table:
        tables.append(table)
    out: list[tuple[float, float]] = []
    for table in tables:
        rows = [r for r in table if RATIO_CELL.fullmatch(r[-1])]
        for i in range(min((len(r) for r in rows), default=0)):
            if all(RATIO_CELL.fullmatch(r[i]) for r in rows):
                values = [float(r[i]) for r in rows]
                out.append((round(min(values), 2), round(max(values), 2)))
    return out


def ratio_endpoint_matches(quoted: str, value: float) -> bool:
    """Whether a quoted endpoint equals a table value at the precision the quote carries.

    A one-decimal quote is compared with the table value rounded to one decimal - half-up, as a
    human rounds it - so `0.9–1.0` certifies a 0.92–0.99 table and `0.87–1.1` does not. Comparing
    with a fixed 0.005 tolerance instead would let `0.87` through against `0.92` (exactly 0.05 apart
    at one decimal), which is the band this gate exists to catch.
    """
    quantum = Decimal(1).scaleb(-len(quoted.partition(".")[2]))
    return Decimal(repr(value)).quantize(quantum, rounding = ROUND_HALF_UP) == Decimal(quoted)


def table_ratio_ranges(text: str) -> list[str]:
    """Ratio ranges the report's own tables print, as written, in document order.

    A table whose ratio cells are ranges (`pp range`, `tg range`) is a ratio table the min/max
    comparison cannot read, and a quote against it fails for that reason - not because the report has
    no table. The two are different problems (edit the quote vs regenerate the evidence), so the
    failure message has to name the right one and the ranges it did find.
    """
    out: list[str] = []
    for line in text.splitlines():
        out += [c for c in markdown_cells(line) if RATIO_RANGE_CELL.fullmatch(c)]
    return out


def percentage_gate(text: str) -> bool:
    """A quoted % or ratio band must be checkable against the report's own table.

    The old gate printed PASS for any `%` in the report without reading a number, which certifies
    arithmetic it never did. Now every number is either checked against the table printed beside it
    or labelled INFO - never PASS.
    """
    good = True
    bands = ratio_columns(text)
    checked: list[tuple[int, int]] = []

    for m in SYMMETRIC_PCT.finditer(text):
        checked.append(m.span())
        good &= bad(f"`{m.group(0).strip()}` is a symmetric band; this project reports one-sided "
                    "ratio intervals (C5) - point at the ratio table's min/max instead")

    for m in RATIO_INTERVAL.finditer(text):
        quoted = m.group(0).strip()
        lo, hi = float(m.group(1)), float(m.group(2))
        if max(lo, hi) > RATIO_SCALE:
            good &= info(f"`{quoted}` is not a ratio band (an endpoint is above {RATIO_SCALE}) - "
                         "not checked against the ratio table")
            continue
        checked.append(m.span())
        if not bands:
            # two different failures wear the same message today: a report with no ratio table, and
            # a report whose ratio table holds ranges where this gate wants a column min/max (T05).
            # The verdict is the same (a quote nothing certifies is not evidence) - the remedy is not
            ranges = table_ratio_ranges(text)
            if not ranges:
                good &= bad(f"`{quoted}`: quote without a table - no ratio table in the report; "
                            "regenerate with `bench/ratios.py` and quote the range it prints")
            elif quoted in ranges:
                # the report's own range cell, read as a quote: naming it "no such range" beside a
                # list that starts with itself would be a message contradicting itself
                good &= bad(f"`{quoted}` is a range the report's own table prints, not a band "
                            "quoted in prose - this gate compares a quote with a ratio column's "
                            "min/max")
            else:
                listed = ", ".join(ranges[:4])
                if len(ranges) > 4:
                    listed += f" (+{len(ranges) - 4} more)"
                good &= bad(f"`{quoted}` matches no range in the report's table(s) (ranges: "
                            f"{listed}): this gate compares a quote with a ratio column's min/max, "
                            "and those cells are ranges")
            continue
        if any(ratio_endpoint_matches(m.group(1), a) and ratio_endpoint_matches(m.group(2), b)
               for a, b in bands):
            good &= ok(f"`{quoted}` matches this report's own ratio table")
        else:
            a, b = min(bands, key=lambda band: abs(lo - band[0]) + abs(hi - band[1]))
            good &= bad(f"`{quoted}` does not match this report's ratio table: the table's ratios "
                        f"run {a:.2f} to {b:.2f}")

    for m in PCT.finditer(text):
        if any(start <= m.start() < end for start, end in checked):
            continue
        good &= info(f"`{m.group(0).strip()}` is prose, not a ratio band - not checked")
    return good


def post(args) -> bool:
    good = True
    if not args.report or not os.path.exists(args.report):
        return bad(f"report not found: {args.report}")
    with open(args.report, encoding="utf-8", errors="replace") as f:
        text = f.read()
    cited = sorted(set(RUN_ID.findall(text)))
    if not cited:
        return bad("the report cites no run ids: nothing here is checkable")

    rows = {}
    csv_path = os.path.join(args.results, "passes.csv")
    with open(csv_path, newline="", encoding="ascii", errors="replace") as f:
        for r in csv.DictReader(f):
            rows[r.get("run_id", "")] = r

    arms, arm_source = arm_runs(args.results)
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
        good &= warm_gate(os.path.join(args.logs, f"perf-{run}.log"), run, args.allow_cold)
        if row_runner(r) == "spec_decode":
            good &= arm_gate(run, arms, arm_source)

    good &= percentage_gate(text)
    return good


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", action="store_true")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--report", default="", help="report file to verify (post)")
    ap.add_argument("--state", default=os.path.join(HERE, "windows", "state.json"))
    ap.add_argument("--deployed", default=os.path.dirname(RESULTS_DEFAULT),
                    help="directory holding the deployed state.json (default: the results parent, "
                         "_agent/v165; '' disables the comparison)")
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
