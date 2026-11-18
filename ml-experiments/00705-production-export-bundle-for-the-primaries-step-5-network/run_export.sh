#!/bin/bash
# usage: run_export.sh ARM MONTH   ARM is replay, proxy, or drop (experiment 704's arms; replay is experiment 677's own chain).
# Env: DATA_PATH_INTERNAL (required), ORIGINS (fixture origin dates; default is the set used for 2026-04).
# 1. e00705_capture.py reruns the prediction pass of the stored models on the GPU (one process, through
#    experiment 677's gpu.sh, 14 GB host cap) in a scratch root; nothing under another experiment's folder is written.
# 2. e00705_export.py (CPU) writes the bundle; 3. e00705_check.py reloads it in a clean interpreter.
set -e
D="$(cd "$(dirname "$0")" && pwd)"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
ARM=$1; M=$2
B=$DATA_PATH_INTERNAL/ml-experiments
X=$(echo $B/00705-*)
case $ARM in
  replay)
    R=$(echo $B/00677-*/)
    CHAIN=${R}chains/5/$M/work; STATS=${R}chains/3b/$M/work
    BACKTEST="--backtest ${R}backtests/step5w_with_injected_sp1.parquet" ;;
  proxy|drop)
    R=$(echo $B/00704-*/)arms/$ARM/
    CHAIN=${R}chains/5/$M/work; STATS=$CHAIN; BACKTEST="" ;;
  *) echo "bad arm $ARM"; exit 1 ;;
esac
if [ -z "$ORIGINS" ]; then
  [ "$M" = 2026-04 ] || { echo "set ORIGINS for month $M"; exit 1; }
  ORIGINS="2026-04-01 2026-04-03 2026-04-04 2026-04-17 2026-04-30"
fi
S=$X/scratch/${ARM}_$M
mkdir -p $S $X/logs
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
"$D"/../00677-*/gpu.sh ../$(basename "$D")/e00705_capture.py --arm $ARM --month $M --origins $ORIGINS \
  --source-root ${R%/} --chain-work $CHAIN --stats-work $STATS --scratch $S --out $S/dump.npz > $X/logs/capture_${ARM}_$M.log 2>&1
cd "$D/../.."
source ml-experiments/00524*/env.sh
.venv/bin/python "$D"/e00705_export.py --arm $ARM --month $M --dump $S/dump.npz --out $X/bundles/$ARM/$M \
  --source-root ${R%/} --chain-work $CHAIN --stats-work $STATS $BACKTEST
.venv/bin/python "$D"/e00705_check.py $X/bundles/$ARM/$M
