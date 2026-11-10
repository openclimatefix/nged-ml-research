#!/bin/bash
# usage: run.sh <args to e00538_fit.py>; one GPU process, 8 GB host cap, 1 OMP thread
R=/home/jack/dev/nged-ml-research/.claude/worktrees/lag-recent-level-538
cd $R
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
source ml-experiments/00524*/env.sh
exec systemd-run --user --scope -q -p MemoryMax=8G -p OOMPolicy=continue env OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" DATA_PATH_INTERNAL=$DATA_PATH_INTERNAL NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x $R/.venv/bin/python /home/jack/dev/nged-substation-forecast/data/ml-experiments/00538-l1d-plus-recent-level-and-lags/code/e00538_fit.py "$@"
