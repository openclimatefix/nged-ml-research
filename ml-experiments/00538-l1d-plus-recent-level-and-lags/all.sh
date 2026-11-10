#!/bin/bash
D=${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}/ml-experiments/00538-l1d-plus-recent-level-and-lags/code
for arm in R1 R9; do
  $D/run.sh fit --arm $arm && $D/run.sh assemble --arm $arm || exit 1
done
for arm in R1 R9; do $D/run.sh fit --arm $arm --stress --seeds 0 || exit 1; done
echo ALL DONE
