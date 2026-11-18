#!/bin/bash
# usage: gpu.sh SCRIPT [args]: one GPU process, 14 GB host cap, 1 OMP thread (as experiment 677's gpu.sh).
# SCRIPT is a file name in this folder, or an absolute path (the global fits run experiment 677's script).
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
cd "$R" || exit 1
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
export DATA_PATH_INTERNAL
source ml-experiments/00524*/env.sh
SCRIPT=$1; shift
case $SCRIPT in /*) ;; *) SCRIPT=$D/$SCRIPT ;; esac
exec systemd-run --user --scope -q -p MemoryMax=14G -p OOMPolicy=continue env OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" DATA_PATH_INTERNAL="$DATA_PATH_INTERNAL" NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x "$R/.venv/bin/python" "$SCRIPT" "$@"
