#!/bin/bash
# Build the thermal-lag tables, then pretrain, step 2, fit, assemble for each arm and group; one GPU
# process at a time. Arm order: lag-enabled with B1, lag-free with B1, lag-free without B1.
cd "$(cd "$(dirname "$0")" && pwd)"
[ -f "$DATA_PATH_INTERNAL/ml-experiments/00622-bsp-gsp-lagged-against-lag-free-matched-d5-thermal-lag-rematch/trailing/D13.parquet" ] \
  || ./run.sh e00622_b1.py build || exit 1
for arm in "on on" "off on" "off off"; do
  set -- $arm
  for g in bsp gsp; do
    for cmd in pretrain-wt step2 fit assemble; do
      ./run.sh e00622_fit.py $cmd --group $g --lags $1 --b1 $2 --seeds 0 1 || exit 1
    done
  done
done
echo ALL DONE
