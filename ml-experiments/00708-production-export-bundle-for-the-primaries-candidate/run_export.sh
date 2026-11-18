#!/bin/bash
# usage: run_export.sh MONTH   (experiment 704's arm `drop`, refitted by `chain667.sh drop MONTH`)
# Env: DATA_PATH_INTERNAL (required), ALLOW_DIRTY (set for a test bundle from uncommitted code), ORIGINS (fixture origin dates; default is the set used for 2026-10).
# 1. e00708_capture.py reruns the prediction pass of the stored Candidate (K1's lags-masked table T, then the
#    Candidate) on the GPU (one process, through experiment 667's gpu.sh, 8 GB host cap) in a scratch root;
#    nothing under another experiment's folder is written.
# 2. e00708_export.py (CPU) writes the bundle; 3. e00708_check.py reloads it in a clean interpreter.
set -e
D="$(cd "$(dirname "$0")" && pwd)"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
M=$1
B=$DATA_PATH_INTERNAL/ml-experiments
X=$(echo $B/00708-*); [ -d "$X" ] || { echo "no 00708 data folder under $B"; exit 1; }
R=$(echo $B/00704-*/)arms/drop; [ -d "$R" ] || { echo "no 00704 arm folder $R"; exit 1; }
CHAIN=$R/chains/$M/work; SRC=$R/P7_D50_$M
for s in 0 1; do [ -f "$R/P7_D50_$M/work/fits/P5/${M}_d0_s$s.parquet" ] || { echo "the chain has not finished for $M"; exit 1; }; done
if [ -z "$ORIGINS" ]; then
  [ "$M" = 2026-10 ] || { echo "set ORIGINS for month $M"; exit 1; }
  ORIGINS="2026-10-01 2026-10-03 2026-10-24 2026-10-25 2026-10-31"
fi
S=$X/scratch/drop_$M
mkdir -p $S $X/logs
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
"$D"/../00667-*/gpu.sh ../$(basename "$D")/e00708_capture.py --month $M --origins $ORIGINS \
  --chain-work $CHAIN --source-root $SRC --scratch $S --out $S/dump.npz > $X/logs/capture_$M.log 2>&1
cd "$D/../.."
source ml-experiments/00524*/env.sh
systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$D"/e00708_export.py --month $M --dump $S/dump.npz --out $X/bundles/drop/$M \
  --source-root $SRC --chain-work $CHAIN --scratch $S ${ALLOW_DIRTY:+--allow-dirty}
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$D"/e00708_check.py $X/bundles/drop/$M
