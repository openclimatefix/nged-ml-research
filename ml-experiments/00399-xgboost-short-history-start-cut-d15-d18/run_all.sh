#!/bin/bash
# Fit all seven arms one after another inside the 3-thread, 8 GB slot (log: work/fit_<arm>.log).
cd "$(dirname "$0")/../.."
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data NGED_S3_BUCKET_URL=x \
  NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x MLFLOW_DISABLE_AGENT_HINT=1
export PYTHONPATH=ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:ml-experiments/00399-xgboost-short-history-start-cut-d15-d18:studies/quick_wins
LOGS=$DATA_PATH_INTERNAL/ml-experiments/00399-xgboost-short-history-start-cut-d15-d18/work
for arm in N0 NC NCZ NCR N3 N6 NS; do
  E399_ARM=$arm systemd-run --user --scope -q -p MemoryMax=8G -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00399-*/e00399_fit.py \
    --arm $arm --cores 3 > "$LOGS/fit_$arm.log" 2>&1
done
echo ALL_DONE > "$LOGS/fit_done.flag"
