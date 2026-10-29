#!/usr/bin/env bash
# Copy the Windows harness's logs and results into the repo, so a pass can be analysed after the
# session that produced it is gone (and so the evidence is versioned alongside the reports).
#
# Run files are uniquely named by the harness (perf-<tag>-<utc>.log), so a copy is append-only: an
# existing file is never rewritten by a later run. The cumulative CSVs are snapshotted with a UTC
# stamp on every archive, so their history survives too.
#
# Usage:
#   bench/windows/archive_logs.sh                    # v165 harness dir -> wiki/plans/v165-reports/artifacts
#   SRC=/mnt/c/somewhere DEST=/somewhere bench/windows/archive_logs.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="${SRC:-/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165}"
DEST="${DEST:-$REPO/wiki/plans/v165-reports/artifacts}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"

mkdir -p "$DEST/logs" "$DEST/results"

copied=0
for f in "$SRC"/logs/*.log; do
  [ -e "$f" ] || continue
  cp -p "$f" "$DEST/logs/"
  copied=$((copied + 1))
done
for f in "$SRC"/results/*.csv "$SRC"/results/BREACH-*.txt; do
  [ -e "$f" ] || continue
  cp -p "$f" "$DEST/results/"
  copied=$((copied + 1))
done

# Cumulative tables also get a stamped snapshot, so the repo keeps their evolution.
for csv in passes pp tg; do
  [ -f "$SRC/results/$csv.csv" ] || continue
  cp -p "$SRC/results/$csv.csv" "$DEST/results/$csv-$STAMP.csv"
done

echo "archived $copied run files from $SRC -> $DEST (csv snapshot $STAMP)"
echo "logs: $(ls -1 "$DEST/logs" | wc -l)  results: $(ls -1 "$DEST/results" | wc -l)"
