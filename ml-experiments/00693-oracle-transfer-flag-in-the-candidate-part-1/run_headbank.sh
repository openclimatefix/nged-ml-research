#!/bin/bash
# usage: run_headbank.sh MONTH [PASSES] [SUFFIX]: the head banks (inference only) for one quarterly fold, seeds 0 1 2.
D="$(cd "$(dirname "$0")" && pwd)"
M=$1; PASSES=${2:-48}; SUF=${3:-_headbank}
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
SRC=$DATA_PATH_INTERNAL/ml-experiments/00677-building-up-the-candidate-demand-network-one-component-at-a-time
OUT=$DATA_PATH_INTERNAL/ml-experiments/00693-oracle-transfer-flag-in-the-candidate-part-1
mkdir -p $OUT/logs
export E00677_ROOT=$SRC E00677_MONTH=$M E00677_SPAN=3 CHAIN677=$SRC/chains/5/$M/work STATS677=$SRC/chains/3b/$M/work E00677_PHASE=predict
exec $D/gpu.sh e00693_cand.py P7+D50 headbank --months $M --seeds 0 1 2 --work-suffix _${M}$SUF --synthetic 1 \
  --models-from $SRC/P7_D50_${M}_8w_sp3/work/models/P5 --bank-passes $PASSES
