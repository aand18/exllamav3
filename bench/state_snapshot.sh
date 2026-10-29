#!/bin/sh
# Freeze and inspect the engine's hidden per-run state.
#
# Two caches decide what an arm actually runs against, and both live under XDG_CACHE_HOME:
#   exllamav3/expert_stats/<hash>.json  routed-expert frequency -> hot/cold GPU placement
#   exllamav3/token_stats/<hash>.pt     sparse-head token frequency
# The first run on a model writes them; every later run reads them. Comparing arms across that
# boundary compares placements, not knobs (critique C1/C2). run_pass.ps1 therefore copies a pristine
# MASTER into an ACTIVE directory before each run and points XDG_CACHE_HOME at the ACTIVE one, so
# every arm starts from the same profile.
#
# Env: TABBY_AGENT (harness dir), LIVE (engine cache dir).
#
# Usage:
#   bench/state_snapshot.sh status     # what master holds vs the live user cache
#   bench/state_snapshot.sh snapshot   # adopt the live cache as MASTER (do this after one warm-up pass)
#   bench/state_snapshot.sh show       # print the master's placement summary
set -eu

TABBY_AGENT="${TABBY_AGENT:-/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165}"
MASTER="$TABBY_AGENT/engine-state/master/exllamav3"
ACTIVE="$TABBY_AGENT/engine-state/active/exllamav3"
LIVE="${LIVE:-/mnt/c/Users/yoho/.cache/exllamav3}"

list() {
    for d in "$1/expert_stats" "$1/token_stats"; do
        [ -d "$d" ] || { echo "    (missing) $d"; continue; }
        n=$(find "$d" -type f | wc -l)
        echo "    $n file(s)  $d"
    done
}

case "${1:-status}" in
snapshot)
    if [ ! -d "$LIVE" ]; then
        echo "no live engine cache at $LIVE - run one warm-up pass first" >&2
        exit 1
    fi
    mkdir -p "$MASTER"
    cp -a "$LIVE"/. "$MASTER"/
    echo "master updated from $LIVE:"
    list "$MASTER"
    ;;
status)
    echo "master  $MASTER"; list "$MASTER"
    echo "active  $ACTIVE"; list "$ACTIVE"
    if [ -d "$LIVE" ]; then echo "live    $LIVE"; list "$LIVE"; fi
    ;;
show)
    for f in "$MASTER"/expert_stats/*.json; do
        [ -e "$f" ] || continue
        echo "== $f"
        python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
layers = d.get("layers", {})
print(f"  {len(layers)} layers recorded")
for k in sorted(layers)[:3]:
    v = layers[k]
    print(f"  {k}: {str(v)[:120]}")
PY
    done
    ;;
*)
    echo "usage: $0 {status|snapshot|show}" >&2
    exit 2
    ;;
esac
