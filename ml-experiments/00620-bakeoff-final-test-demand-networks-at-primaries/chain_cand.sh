#!/bin/bash
# usage: chain_cand.sh: the class 2 candidate's tables and fit on the final fold (train phase)
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00620-bakeoff-final-test-demand-networks-at-primaries
export E00620_PHASE=train CHAIN620=$R/chains/k1/work
M="--months 2026-07 --seeds 0 1 2 3 4"
set -e
$D/gpu.sh e00620_cand.py P7+D50 table --work-suffix _e3 $M > $R/logs/cand_table.log 2>&1
$D/gpu.sh e00620_cand.py P7+D50 fit --work-suffix _e3 $M > $R/logs/cand_fit.log 2>&1
