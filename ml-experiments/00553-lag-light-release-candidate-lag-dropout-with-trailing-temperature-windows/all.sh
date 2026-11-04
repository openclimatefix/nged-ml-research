#!/bin/bash
# Fits both dropout rates (5 folds x 3 seeds), assembles them, then the injected-shift runs (seed 0).
D=$(cd "$(dirname "$0")" && pwd)
L=$DATA_PATH_INTERNAL/ml-experiments/00553-lag-light-release-candidate-lag-dropout-with-trailing-temperature-windows/logs
mkdir -p "$L"
"$D/run.sh" e00553_pretrain.py > "$L/pretrain.log" 2>&1 || exit 1
for arm in RDWT-D30 RDWT-D15; do
  "$D/run.sh" e00553_fit.py fit --arm $arm > "$L/fit_$arm.log" 2>&1 && "$D/run.sh" e00553_fit.py assemble --arm $arm >> "$L/fit_$arm.log" 2>&1 || exit 1
done
for arm in RDWT-D30 RDWT-D15; do "$D/run.sh" e00553_fit.py fit --arm $arm --stress --seeds 0 > "$L/stress_$arm.log" 2>&1 || exit 1; done
echo ALL DONE > "$L/done"
