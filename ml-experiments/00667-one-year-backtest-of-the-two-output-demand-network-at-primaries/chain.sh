#!/bin/bash
# usage: chain.sh MONTH [STEP...]: one backtest fold. Steps (default all, in order):
# pretrain, step2 (K1's chain), table, fit (Candidate, train phase: held-out power hidden), then
# ptable, pfit (Candidate, predict phase: the month's window opened, forecasts written).
D="$(cd "$(dirname "$0")" && pwd)"
R=${E00667_ROOT:-$DATA_PATH_INTERNAL/ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries}
export E00667_ROOT=$R
M=$1; shift
STEPS=${*:-pretrain step2 table fit ptable pfit}
export E00667_MONTH=$M CHAIN667=$R/chains/$M/work
mkdir -p $R/logs $CHAIN667
A="--months $M --seeds 0 1"
set -e
for s in $STEPS; do
  case $s in
    pretrain|step2) E00667_PHASE=train $D/gpu.sh e00667_k1.py k1 $s $A > $R/logs/${M}_$s.log 2>&1 ;;
    table|fit) E00667_PHASE=train $D/gpu.sh e00667_cand.py P7+D50 $s --work-suffix _$M $A > $R/logs/${M}_c$s.log 2>&1 ;;
    ptable) E00667_PHASE=predict $D/gpu.sh e00667_cand.py P7+D50 table --work-suffix _$M $A > $R/logs/${M}_c$s.log 2>&1 ;;
    pfit) rm -f $R/P7_D50_$M/work/fits/P5/${M}_d0_s?.empty  # the train phase's markers; fit_arm would skip on them
      E00667_PHASE=predict $D/gpu.sh e00667_cand.py P7+D50 fit --work-suffix _$M $A > $R/logs/${M}_c$s.log 2>&1 ;;
  esac
  echo "$M $s done $(date -u +%T)"
done
