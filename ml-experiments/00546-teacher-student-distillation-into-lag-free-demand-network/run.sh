#!/bin/bash
# usage: run.sh <script name without .py> <args>; one process, 8 GB host cap, 1 OMP thread.
# Run from the worktree root. Puts every experiment folder on PYTHONPATH.
set -e
cd "$(git rev-parse --show-toplevel)"
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
source ml-experiments/00524*/env.sh
script=$1; shift
exec systemd-run --user --scope -q -p MemoryMax=8G -p OOMPolicy=continue \
  env OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" DATA_PATH_INTERNAL="$DATA_PATH_INTERNAL" \
  NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x \
  .venv/bin/python ml-experiments/00546-teacher-student-distillation-into-lag-free-demand-network/$script.py "$@"
