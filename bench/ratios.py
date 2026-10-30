#!/usr/bin/env python3
"""Per-band ratios for the v1.6.5 retune, from the harness CSVs.

The plan's acceptance rule is a per-band ratio against a reference measured in the same session:
median of the ratios must exceed the band's own spread (Task 4). This script does exactly that
arithmetic, and nothing else -- it reads `passes.csv` / `pp.csv` / `tg.csv` and prints a markdown
table, so the numbers in a report can be regenerated from the archived CSVs.

Every run is identified by its `run_id` (tag + UTC stamp); a plain tag is accepted and resolves to
the newest `ok` run of that tag, so a re-run never silently mixes with an older attempt.

Usage:
  # spread of a repeated reference (Task 4): A2 / A1 per band
  bench/ratios.py --dir <results> --ref armA-1 --var armA-2

  # an ABBA comparison: two references, two variants; one ratio per reference/variant pair
  bench/ratios.py --dir <results> --ref memops0-a --var memops1-a \
                              --ref2 memops0-b --var2 memops1-b

  # drafting arms (spec_decode.csv): per-category tok/s + acceptance for two runs
  bench/ratios.py --dir <results> --ref mtp-ref2-318 --var mtp-ndt2-318

  # what ran, newest first
  bench/ratios.py --dir <results> --list

Output: markdown to stdout. Redirect it into the task report (or the artifacts dir) so the
analysis survives the session.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys


def read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="ascii", errors="replace") as f:
        return [r for r in csv.DictReader(f)]


def runs(passes: list[dict]) -> list[dict]:
    return [p for p in passes if p.get("run_id")]


def resolve(passes: list[dict], spec: str) -> dict:
    """Exact run_id, else the newest ok run whose tag or run_id starts with `spec`."""
    exact = [p for p in passes if p.get("run_id") == spec]
    if exact:
        return exact[-1]
    pref = [p for p in passes if (p.get("tag") == spec or (p.get("run_id") or "").startswith(spec))]
    ok = [p for p in pref if p.get("status") == "ok"]
    if ok:
        return ok[-1]
    if pref:
        return pref[-1]
    known = sorted({p.get("tag", "?") for p in passes})
    raise SystemExit(f"no run matches {spec!r}; tags present: {', '.join(known)}")


def pp_bands(pp: list[dict], run_id: str) -> dict[int, float]:
    return {int(r["band"]): float(r["pp_tok_s"]) for r in pp if r.get("run_id") == run_id}


def tg_bands(tg: list[dict], run_id: str) -> dict[int, float]:
    """Generation speed at S=1 per context (the S=2..4 columns are verification cost, not speed)."""
    out = {}
    for r in tg:
        if r.get("run_id") != run_id:
            continue
        try:
            out[int(r["context"])] = float(r["s1"])
        except (KeyError, ValueError):
            continue
    return out


def spec_arms(spec: list[dict], run_id: str) -> dict[str, dict[str, dict]]:
    """category -> arm -> {tok_s, acc, draft}; the drafting table from spec_decode.py."""
    out: dict[str, dict[str, dict]] = {}
    for r in spec:
        if r.get("run_id") != run_id:
            continue
        try:
            cell = {"tok_s": float(r["tok_s"] or 0), "acc": float(r.get("acc") or -1),
                    "draft": float(r.get("draft") or -1)}
        except (TypeError, ValueError):
            continue
        out.setdefault(r.get("category", "?"), {})[r.get("arm", "?")] = cell
    return out


def fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def ratio_table(kind: str, a: dict[int, float], b: dict[int, float]) -> list[str]:
    lines = [f"| {kind} band | reference | variant | ratio (var/ref) |", "|---|---|---|---|"]
    for band in sorted(set(a) & set(b)):
        r = b[band] / a[band] if a[band] else None
        lines.append(f"| {band} | {fmt(a[band])} | {fmt(b[band])} | {fmt(r)} |")
    return lines


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="C:/Users/yoho/Downloads/tabbyAPI/_agent/v165/results",
                    help="results directory holding passes.csv, pp.csv, tg.csv")
    ap.add_argument("--ref", help="reference run_id or tag")
    ap.add_argument("--var", help="variant run_id or tag")
    ap.add_argument("--ref2", help="second reference (ABBA)")
    ap.add_argument("--var2", help="second variant (ABBA)")
    ap.add_argument("--list", action="store_true", help="list runs, newest last")
    args = ap.parse_args(argv)

    d = pathlib.Path(args.dir)
    passes, pp, tg = read_csv(d / "passes.csv"), read_csv(d / "pp.csv"), read_csv(d / "tg.csv")
    spec = read_csv(d / "spec_decode.csv")
    if not passes:
        raise SystemExit(f"no passes.csv in {d}")

    if args.list or not args.ref:
        print("| run_id | tag | status | max_length | split | min_vram | min_ram | runner | args |")
        print("|---|---|---|---|---|---|---|---|---|")
        for p in runs(passes):
            print(f"| {p['run_id']} | {p['tag']} | {p['status']} | {p.get('max_length','')} | "
                  f"{p.get('split','')} | {p.get('min_free_vram_mb','')} | {p.get('min_free_ram_mb','')} | "
                  # rows written before the runner column existed have no value there; every one of
                  # them predates spec_decode support, so "perf" is what they were, not a guess
                  f"{p.get('runner') or 'perf'} | `{(p.get('args') or '')[:70]}` |")
        return 0

    if not args.var:
        raise SystemExit("--var is required with --ref")

    pairs = [(args.ref, args.var)]
    if args.ref2 or args.var2:
        if not (args.ref2 and args.var2):
            raise SystemExit("--ref2 and --var2 must be given together")
        pairs.append((args.ref2, args.var2))

    resolved = []
    for ref_spec, var_spec in pairs:
        ref, var = resolve(passes, ref_spec), resolve(passes, var_spec)
        if ref["status"] != "ok" or var["status"] != "ok":
            print(f"!! {ref['run_id']} status={ref['status']}, {var['run_id']} status={var['status']}"
                  " -- a non-ok pass is not a measurement", file=sys.stderr)
        resolved.append((ref, var))
        print(f"\n### {var['run_id']} vs {ref['run_id']}")
        sr, sv = spec_arms(spec, ref["run_id"]), spec_arms(spec, var["run_id"])
        if sr and sv:
            print("\n| category | reference (t/s) | reference acc | variant (t/s) | variant acc | ratio |")
            print("|---|---|---|---|---|---|")
            ratios = []
            for cat in sorted(set(sr) & set(sv)):
                # the table labels the reference column "Baseline"; compare any non-baseline arm
                ar = next((k for k in sr[cat] if not k.strip().lower().startswith("baseline")), None)
                av = next((k for k in sv[cat] if not k.strip().lower().startswith("baseline")), None)
                if not (ar and av):
                    continue
                a, b = sr[cat][ar], sv[cat][av]
                if not a["tok_s"]:
                    continue
                r = b["tok_s"] / a["tok_s"]
                ratios.append(r)
                fa = "-" if a["acc"] < 0 else f"{a['acc']:.2f}"
                fb = "-" if b["acc"] < 0 else f"{b['acc']:.2f}"
                print(f"| {cat} | {a['tok_s']:.2f} | {fa} | {b['tok_s']:.2f} | {fb} | {r:.3f} |")
            if ratios:
                print(f"\n**median drafting ratio {statistics.median(ratios):.3f}** "
                      f"(range {min(ratios):.3f}-{max(ratios):.3f}, n={len(ratios)})")
                continue
        for kind, getter in (("pp", pp_bands), ("tg (S=1)", tg_bands)):
            a, b = getter(pp if kind == "pp" else tg, ref["run_id"]), getter(pp if kind == "pp" else tg, var["run_id"])
            if not a or not b:
                print(f"\n_no {kind} rows for one side_")
                continue
            print()
            print("\n".join(ratio_table(kind, a, b)))

    if len(resolved) > 1:
        print("\n### combined (median of the two per-band ratios)")
        for kind, getter in (("pp", pp_bands), ("tg (S=1)", tg_bands)):
            src = pp if kind == "pp" else tg
            per_band: dict[int, list[float]] = {}
            for ref, var in resolved:
                a, b = getter(src, ref["run_id"]), getter(src, var["run_id"])
                for band in set(a) & set(b):
                    if a[band]:
                        per_band.setdefault(band, []).append(b[band] / a[band])
            if not per_band:
                continue
            print(f"\n| {kind} band | median ratio | min | max | spread (max-min) |")
            print("|---|---|---|---|---|")
            for band in sorted(per_band):
                rs = per_band[band]
                print(f"| {band} | {statistics.median(rs):.3f} | {min(rs):.3f} | {max(rs):.3f} | "
                      f"{max(rs) - min(rs):.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
