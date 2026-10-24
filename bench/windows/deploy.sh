#!/usr/bin/env bash
# Deploy the Windows measurement harness from the repo to the directory the box runs it from.
#
# The repo is the source of truth: bench/windows/*.ps1 and eval/perf.py. The Windows copy under
# C:/Users/yoho/Downloads/tabbyAPI/_agent/v164 is a deployed artifact; editing it there is how two
# benchmarks end up incomparable.
#
# Usage:
#   bench/windows/deploy.sh                 # deploy to the default directory
#   DEST=/mnt/c/somewhere bench/windows/deploy.sh
#
# Then run a pass (from WSL; PowerShell needs the Windows path):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
#     "C:\Users\yoho\Downloads\tabbyAPI\_agent\v164\run_pass.ps1" -Tag armA-a -MaxLength 262144
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DEST="${DEST:-/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v164}"

mkdir -p "$DEST/logs" "$DEST/results"

install -m 0666 "$HERE/watch.ps1"      "$DEST/watch.ps1"
install -m 0666 "$HERE/run_pass.ps1"   "$DEST/run_pass.ps1"
install -m 0666 "$HERE/run_build.ps1"  "$DEST/run_build.ps1"
# perf.py is a verbatim copy of the tree's eval/perf.py (identical in v1.5.4 and v1.6.4); it is
# copied rather than committed twice. It imports the installed engine, not this tree.
install -m 0666 "$REPO/eval/perf.py"   "$DEST/perf.py"

echo "deployed to $DEST"
sha256sum "$DEST"/*.ps1 "$DEST"/perf.py
