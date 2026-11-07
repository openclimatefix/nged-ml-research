#!/bin/bash
# usage: predict.sh: open the final-test window ONCE. Run only after the manager's "go".
# Verifies the freeze, creates the lock (a second run is refused), forecasts all three networks, assembles.
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00620-bakeoff-final-test-demand-networks-at-primaries
set -e
$D/gpu.sh e00620_freeze.py --verify
set -o noclobber
echo "started $(date -u +%FT%TZ)" > $D/final_test_620.lock
set +o noclobber
export E00620_PHASE=predict
M="--months 2026-07 --seeds 0 1 2 3 4"
CHAIN620=$R/chains/k1/work $D/gpu.sh e00620_k1.py k1 fit $M > $R/logs/final_k1.log 2>&1
CHAIN620=$R/chains/k3/work $D/gpu.sh e00620_k3.py k3 fit $M > $R/logs/final_k3.log 2>&1
export CHAIN620=$R/chains/k1/work
$D/gpu.sh e00620_cand.py P7+D50 table --work-suffix _e3 $M > $R/logs/final_cand_table.log 2>&1
$D/gpu.sh e00620_cand.py P7+D50 fit --work-suffix _e3 $M > $R/logs/final_cand_fit.log 2>&1
$D/gpu.sh e00620_assemble.py 2026-07 > $R/logs/final_assemble.log 2>&1
echo "finished $(date -u +%FT%TZ)" >> $D/final_test_620.lock
