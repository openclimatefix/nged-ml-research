#!/bin/bash
# usage: bank.sh MONTH...: injected-shift passes (12 passes, 2 seeds) of the Candidate fold, then the head rows. One GPU process at a time.
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries
mkdir -p $R/prevailing/logs
for M in "$@"; do
  E00667_MONTH=$M CHAIN667=$R/chains/$M/work E00667_PHASE=predict \
    $D/gpu_p.sh p667_inject.py P7+D50 inject --work-suffix _$M --months $M --seeds 0 1 > $R/prevailing/logs/inject_$M.log 2>&1
  OMP_NUM_THREADS=2 $D/cpu_p.sh p667_data.py $M > $R/prevailing/logs/data_$M.log 2>&1
  echo "$M bank done $(date -u +%T)"
done
