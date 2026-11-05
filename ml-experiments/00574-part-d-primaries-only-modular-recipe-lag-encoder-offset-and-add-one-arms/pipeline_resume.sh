#!/bin/bash
set -e
D="$(cd "$(dirname "$0")" && pwd)"
for arm in P2 P2plac; do
  $D/run.sh pretrain-wt --arm $arm
  $D/run.sh fit --arm $arm
  $D/run.sh fit --arm $arm --seeds 0 --stress
  $D/run.sh assemble --arm $arm
done
echo PIPELINE DONE
