#!/bin/bash
D=/home/jack/dev/nged-substation-forecast/data/ml-experiments/00539-l1d-plus-day-and-week-lags/code
for arm in R-DW; do
  $D/run.sh fit --arm $arm && $D/run.sh assemble --arm $arm || exit 1
done
for arm in R-DW; do $D/run.sh fit --arm $arm --stress --seeds 0 || exit 1; done
echo ALL DONE
