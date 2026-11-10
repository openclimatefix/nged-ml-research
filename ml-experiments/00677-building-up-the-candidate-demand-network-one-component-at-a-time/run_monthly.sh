#!/bin/bash
# usage: run_monthly.sh; monthly refits (span 1) of steps 5 to 8, seeds 0 to 2, twelve folds one after the other.
D="$(cd "$(dirname "$0")" && pwd)"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
export SEEDS="0 1 2"
set -e
for m in 2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06 2026-07 2026-08 2026-09 2026-10; do
  for s in 5w 5wo 6w 6wo 7w 7wo 8w 8wo; do
    $D/chain.sh $m 1 $s
  done
done
echo "MONTHLY FINISHED $(date -u +%T)"
