"""VRAM guard watchdog (runs WSL-side): kill a benchmark task if GPU free < threshold.

Any GPU run (VRAM is a factor) must be wrapped with this guard: it polls
``nvidia-smi memory.free`` and ``taskkill``s the task the moment free VRAM
drops under ``--min-free-mb`` (default 100). Only PIDs that appeared AFTER
``--before`` are ever killed, so pre-existing processes are never touched.

Workflow per GPU run (bash, WSL)::

    BEFORE=$(cmd.exe /c tasklist | awk '$1=="python.exe"{print $2}' | paste -sd,)
    cmd.exe /c "cd /d <dir>&& call run_foo.bat" > launcher.log 2>&1 &
    ./venv/bin/python eval/smi_guard.py --image python.exe --before "$BEFORE"

Exit codes: 0 = task finished, GPU idle; 2 = KILLED (free < min);
3 = timeout (task killed, keeps GPU idle); 4 = no task ever appeared.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

IMAGE_COL_WIDTH = 25  # tasklist pads the image name; PID is the next token


def task_pids(image: str) -> set[int]:
    """PIDs of a Windows image via tasklist (parsed WSL-side)."""
    out = subprocess.run(["cmd.exe", "/c", "tasklist"], capture_output=True,
                         text=True, timeout=60).stdout
    pids: set[int] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() == image.lower() \
                and parts[1].isdigit():
            pids.add(int(parts[1]))
    return pids


def min_free_mb() -> int:
    """Minimum free VRAM across GPUs, MiB."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60).stdout
    vals = [int(x) for x in out.replace(",", " ").split() if x.isdigit()]
    if not vals:
        raise RuntimeError("nvidia-smi returned no memory.free values")
    return min(vals)


def kill(pids: set[int]) -> None:
    for pid in sorted(pids):
        r = subprocess.run(["taskkill.exe", "/F", "/PID", str(pid)],
                           capture_output=True, text=True, timeout=60)
        print(f"taskkill {pid}: {r.stdout.strip() or r.stderr.strip()}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True,
                    help="Windows image name, e.g. python.exe")
    ap.add_argument("--before", default="",
                    help="CSV of pre-existing PIDs to never touch")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--grace", type=int, default=30,
                    help="Seconds to wait for the task to appear")
    ap.add_argument("--min-free-mb", type=int, default=100)
    a = ap.parse_args()
    before = {int(x) for x in a.before.replace(",", " ").split()
              if x.isdigit()}
    t0 = time.time()
    seen = False
    while True:
        now = time.time()
        targets = task_pids(a.image) - before
        if targets:
            seen = True
        if seen and not targets:
            print(f"DONE image={a.image} free={min_free_mb()}MiB", flush=True)
            return 0
        if not seen and now - t0 > a.grace:
            print(f"NO-TASK image={a.image} (nothing new in {a.grace}s)",
                  flush=True)
            return 4
        if now - t0 > a.timeout:
            print(f"TIMEOUT {int(now - t0)}s, killing {sorted(targets)}",
                  flush=True)
            kill(targets)
            return 3
        try:
            free = min_free_mb()
        except RuntimeError as e:
            print(f"smi error: {e}", flush=True)
            time.sleep(a.interval)
            continue
        if int(now - t0) % 30 < a.interval:
            print(f"watch {a.image} pids={sorted(targets)} "
                  f"free={free}MiB t={int(now - t0)}s", flush=True)
        if free < a.min_free_mb and targets:
            print(f"KILL free={free}MiB < {a.min_free_mb}MiB "
                  f"pids={sorted(targets)}", flush=True)
            kill(targets)
            return 2
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
