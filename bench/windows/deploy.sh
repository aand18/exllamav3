#!/usr/bin/env bash
# Deploy the Windows measurement harness from the repo to the directory the box runs it from.
#
# The repo is the source of truth: bench/windows/*.ps1 and eval/perf.py. The Windows copy under
# C:/Users/yoho/Downloads/tabbyAPI/_agent/v165 is a deployed artifact; editing it there is how two
# benchmarks end up incomparable.
#
# Usage:
#   bench/windows/deploy.sh                 # deploy to the default directory
#   DEST=/mnt/c/somewhere bench/windows/deploy.sh
#
# Then run a pass (from WSL; PowerShell needs the Windows path):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
#     "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag armA-a
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DEST="${DEST:-/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165}"

mkdir -p "$DEST/logs" "$DEST/results"

# The active regime lives in state.json; run_pass.ps1 refuses to start without it. Ship the repo's
# copy only when the destination has none, so a swap recorded on the box is never overwritten.
if [ ! -f "$DEST/state.json" ]; then
  install -m 0666 "$HERE/state.json" "$DEST/state.json"
fi

install -m 0666 "$HERE/watch.ps1"      "$DEST/watch.ps1"
install -m 0666 "$HERE/run_pass.ps1"   "$DEST/run_pass.ps1"
install -m 0666 "$HERE/run_build.ps1"  "$DEST/run_build.ps1"
# perf.py is a verbatim copy of the tree's eval/perf.py (identical in v1.5.4 and v1.6.4); it is
# copied rather than committed twice. It imports the installed engine, not this tree.
install -m 0666 "$REPO/eval/perf.py"   "$DEST/perf.py"
# T7 drafter arms: -or override specs that shadow the MTP head from a sidecar file, no base-dir copy.
install -m 0666 "$HERE"/mtp-sidecar-*.yaml "$DEST"/
# T6/T7 drafting arms run through run_pass.ps1 -Runner spec_decode; the script reads its prompt
# corpus from ./prompts next to itself, so both travel together or every run dies at the first
# workload (measured 2026-10-29).
install -m 0666 "$REPO/eval/spec_decode.py" "$DEST/spec_decode.py"
mkdir -p "$DEST/prompts"
cp -f "$REPO"/eval/prompts/*.json "$DEST/prompts/"

echo "deployed to $DEST"
sha256sum "$DEST"/*.ps1 "$DEST"/perf.py
