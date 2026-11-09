#!/bin/bash
# usage: gpu.sh SCRIPT [args]; one GPU process, 16 GB host cap, 1 OMP thread
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
cd $R
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
export DATA_PATH_INTERNAL MLFLOW_DISABLE_AGENT_HINT=1
source ml-experiments/00524*/env.sh
exec systemd-run --user --scope -q -p MemoryMax=16G -p OOMPolicy=continue env OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" E00669_ROOT="${E00669_ROOT:-}" DATA_PATH_INTERNAL=$DATA_PATH_INTERNAL MLFLOW_DISABLE_AGENT_HINT=1 E00669_HIDE="${E00669_HIDE:-0}" NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x $R/.venv/bin/python $D/"$@"
