#!/bin/bash
# usage: run.sh <script.py> <args>; one GPU process, 8 GB host cap, 1 OMP thread.
# Needs DATA_PATH_INTERNAL; run from anywhere inside the repository checkout.
R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
source ml-experiments/00524*/env.sh
S=$1; shift
exec systemd-run --user --scope -q -p MemoryMax=8G -p OOMPolicy=continue env OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" DATA_PATH_INTERNAL=$DATA_PATH_INTERNAL NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x "$R/.venv/bin/python" "$R/ml-experiments/00553-lag-light-release-candidate-lag-dropout-with-trailing-temperature-windows/$S" "$@"
