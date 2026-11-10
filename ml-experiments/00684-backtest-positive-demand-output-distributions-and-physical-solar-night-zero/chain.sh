#!/bin/bash
# usage: chain.sh FAMILY MONTH [STEP...]: experiment 667's chain for one fold under an output family
# (FAMILY gauss, tgauss, or gamma). The encoder pre-training is experiment 667's (linked, not
# re-run); steps default to: step2 table fit ptable pfit. One GPU process at a time.
D="$(cd "$(dirname "$0")" && pwd)"
F667=$(ls -d "$D"/../00667-*)
SRC=${E00667_SRC:-$DATA_PATH_INTERNAL/ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries}
R=$DATA_PATH_INTERNAL/ml-experiments/00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/demand_$1
export E00667_ROOT=$R E00684_FAMILY=$1
shift; M=$1; shift
STEPS=${*:-step2 table fit ptable pfit}
export E00667_MONTH=$M CHAIN667=$R/chains/$M/work
mkdir -p $R/logs $CHAIN667/logs
for l in pretrained_P2 stats; do [ -e $CHAIN667/$l ] || ln -s $SRC/chains/$M/work/$l $CHAIN667/$l; done
A="--months $M --seeds 0 1"
set -e
run() { "$F667/gpu.sh" ../$(basename "$D")/e00684_run.py "$@"; }
for s in $STEPS; do
  case $s in
    step2) E00667_PHASE=train run e00667_k1.py k1 $s $A > $R/logs/${M}_$s.log 2>&1 ;;
    table|fit) E00667_PHASE=train run e00667_cand.py P7+D50 $s --work-suffix _$M $A > $R/logs/${M}_c$s.log 2>&1 ;;
    ptable) E00667_PHASE=predict run e00667_cand.py P7+D50 table --work-suffix _$M $A > $R/logs/${M}_c$s.log 2>&1 ;;
    pfit) rm -f $R/P7_D50_$M/work/fits/P5/${M}_d0_s?.empty
      E00667_PHASE=predict run e00667_cand.py P7+D50 fit --work-suffix _$M $A > $R/logs/${M}_c$s.log 2>&1 ;;
  esac
  echo "$E00684_FAMILY $M $s done $(date -u +%T)"
done
