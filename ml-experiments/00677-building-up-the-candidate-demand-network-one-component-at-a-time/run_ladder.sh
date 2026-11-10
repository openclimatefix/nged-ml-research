#!/bin/bash
# usage: run_ladder.sh PHASE; PHASE is "seeds012" (every step, seeds 0 to 2) or "seeds34" (seeds 3 and 4 for steps 5 to 8).
# Quarterly folds, one after the other; one GPU process at a time. Skips nothing by itself: the fit scripts skip finished cells.
D="$(cd "$(dirname "$0")" && pwd)"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
FOLDS="2025-11 2026-02 2026-05 2026-08"
set -e
case $1 in
  seeds012) export SEEDS="0 1 2"; STEPS="3b 1 2 3a 4 5w 5wo 6w 6wo 7w 7wo 8w 8wo" ;;
  seeds34) export SEEDS="3 4"; STEPS="5w 5wo 6w 6wo 7w 7wo 8w 8wo" ;;
  *) echo "bad phase"; exit 1 ;;
esac
for m in $FOLDS; do
  for s in $STEPS; do
    $D/chain.sh $m 3 $s
  done
done
echo "PHASE $1 FINISHED $(date -u +%T)"
