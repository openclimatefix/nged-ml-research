#!/bin/bash
# usage: DATA_PATH_INTERNAL=<private data folder> run.sh <script> <args>; one GPU process, 8 GB host cap, 1 OMP thread.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source ml-experiments/00552*/env.sh
S=$1; shift
exec systemd-run --user --scope -q -p MemoryMax=8G -p AllowedCPUs=0-3 -p OOMPolicy=continue env OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" DATA_PATH_INTERNAL=$DATA_PATH_INTERNAL NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x .venv/bin/python "$S" "$@"
