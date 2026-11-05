#!/bin/bash
# Part D add-one pipeline: one GPU process at a time. Stage 1: arms that reuse 572's pre-training.
set -e
D="$(cd "$(dirname "$0")" && pwd)"
for arm in P1 P3E0; do
  $D/run.sh fit --arm $arm
  $D/run.sh fit --arm $arm --seeds 0 --stress
  $D/run.sh assemble --arm $arm
done
for arm in P4 P4E; do
  $D/run.sh step2 --arm $arm
  $D/run.sh fit --arm $arm
  $D/run.sh fit --arm $arm --seeds 0 --stress
  $D/run.sh assemble --arm $arm
done
for arm in P2 P2plac; do
  $D/run.sh pretrain-wt --arm $arm
  $D/run.sh fit --arm $arm
  $D/run.sh fit --arm $arm --seeds 0 --stress
  $D/run.sh assemble --arm $arm
done
echo PIPELINE DONE
