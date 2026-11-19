#!/bin/bash
# Phase E copy of experiment 667's chain.sh for the months 2025-04 to 2025-06.
# usage: e00709_e_chain.sh MONTH [STEP...]: one backtest fold. Steps (default pretrain step2 table ptable):
# pretrain, step2 (K1's chain), table, then
# ptable, pfit (Candidate, predict phase: the month's window opened, forecasts written).
D="$(cd "$(dirname "$0")" && pwd)"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
R=$DATA_PATH_INTERNAL/ml-experiments/00709-era-signal-in-the-candidate-part-1/e_667  # phase E: nothing is written into experiment 667's folder
export E00667_ROOT=$R
M=$1; shift
STUBS=$DATA_PATH_INTERNAL/ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/inputs/frames
STUBS_BEFORE=$(ls $STUBS | wc -l)  # a run with no demand frame makes fold.install write a stub here: that must not happen
STEPS=${*:-pretrain step2 table ptable}
export E00667_MONTH=$M CHAIN667=$R/chains/$M/work
mkdir -p $R/logs $CHAIN667
A="--months $M --seeds 0 1"
set -e
for s in $STEPS; do
  case $s in
    pretrain|step2) E00667_PHASE=train $D/e00709_e_gpu.sh e00709_e_k1.py k1 $s $A > $R/logs/${M}_$s.log 2>&1 ;;
    table|fit) E00667_PHASE=train $D/e00709_e_gpu.sh e00709_e_cand.py P7+D50 $s --work-suffix _$M $A > $R/logs/${M}_c$s.log 2>&1 ;;
    ptable) E00667_PHASE=predict $D/e00709_e_gpu.sh e00709_e_cand.py P7+D50 table --work-suffix _$M $A > $R/logs/${M}_c$s.log 2>&1 ;;
    pfit) rm -f $R/P7_D50_$M/work/fits/P5/${M}_d0_s?.empty  # the train phase's markers; fit_arm would skip on them
      E00667_PHASE=predict $D/e00709_e_gpu.sh e00709_e_cand.py P7+D50 fit --work-suffix _$M $A > $R/logs/${M}_c$s.log 2>&1 ;;
  esac
  echo "$M $s done $(date -u +%T)"
done
[ "$(ls $STUBS | wc -l)" = "$STUBS_BEFORE" ] || { echo "ERROR: stub frames were written into experiment 667 inputs/frames"; exit 1; }
