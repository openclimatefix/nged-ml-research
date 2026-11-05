#!/bin/bash
# Part 1 pipeline: pre-train the two new lag arms, fit every arm (coarse, seeds 0 and 1), stress fits (seed 0), assemble.
set -e
D="$(cd "$(dirname "$0")" && pwd)"
for arm in V3D5 V35; do $D/run.sh pretrain-wt --arm $arm; done
for arm in E2ft V3 V3D5 V35; do
  $D/run.sh fit --arm $arm
  $D/run.sh fit --arm $arm --seeds 0 --stress
  $D/run.sh assemble --arm $arm
done
echo PIPELINE DONE
