#!/bin/bash
D=$(cd "$(dirname "$0")" && pwd)
for arm in R-MED R-DROP; do
  "$D/run.sh" fit --arm $arm && "$D/run.sh" assemble --arm $arm || exit 1
done
for arm in R-MED R-DROP; do "$D/run.sh" fit --arm $arm --stress --seeds 0 || exit 1; done
echo ALL DONE
