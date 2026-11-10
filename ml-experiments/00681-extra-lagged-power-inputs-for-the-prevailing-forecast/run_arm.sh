#!/bin/bash
# usage: run_arm.sh ARM [WIDTH] [MONTHS...]: fit one arm on the CPU for the development folds, 1 thread, 16 GB cap.
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
cd "$R"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
source ml-experiments/00524*/env.sh
ARM=$1; WIDTH=${2:-0}; shift 2 2>/dev/null
MONTHS=${@:-2026-01 2026-02 2026-03 2026-04 2026-05 2026-06}
for M in $MONTHS; do
  systemd-run --user --scope -q -p MemoryMax=16G -p OOMPolicy=continue env OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" \
    DATA_PATH_INTERNAL=$DATA_PATH_INTERNAL CUDA_VISIBLE_DEVICES="" $R/.venv/bin/python $D/e00681_fit.py $ARM $M --width $WIDTH 2>&1 | grep -v mlflow || exit 1
done
