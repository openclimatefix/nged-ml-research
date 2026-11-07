#!/bin/bash
# usage: chain_rest.sh: K3 pre-training and fit, then the class 2 candidate's tables and fit (train phase), after K1's chain
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00620-bakeoff-final-test-demand-networks-at-primaries
export E00620_PHASE=train
while pgrep -f chain_k1.sh >/dev/null; do sleep 10; done
M="--months 2026-07 --seeds 0 1 2 3 4"
CHAIN620=$R/chains/k3/work $D/gpu.sh e00620_k3.py k3 pretrain $M > $R/logs/k3_pretrain.log 2>&1
CHAIN620=$R/chains/k3/work $D/gpu.sh e00620_k3.py k3 fit $M > $R/logs/k3_fit.log 2>&1
export CHAIN620=$R/chains/k1/work
$D/gpu.sh e00620_cand.py P7+D50 table --work-suffix _e3 $M > $R/logs/cand_table.log 2>&1
$D/gpu.sh e00620_cand.py P7+D50 fit --work-suffix _e3 $M > $R/logs/cand_fit.log 2>&1
