#!/bin/bash
# usage: watch.sh: build the bank of every fold month as soon as the 667 worker's Candidate fit of that month has finished (both seeds, predict phase).
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries
MONTHS="2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06 2026-07 2026-08 2026-09 2026-10"
while true; do
  left=0
  for M in $MONTHS; do
    [ -f $R/prevailing/head_data/bank_$M.npz ] && continue
    left=1
    if grep -q "CHECKS PASSED" $R/logs/${M}_cpfit.log 2>/dev/null && [ -f $R/P7_D50_$M/work/fits/P5/${M}_d0_s1.parquet ]; then
      $D/bank.sh $M
    fi
  done
  [ $left = 0 ] && break
  sleep 60
done
echo "all banks built $(date -u +%T)"
