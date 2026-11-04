#!/bin/bash
D=/home/jack/dev/nged-ml-research/.claude/worktrees/lag-light-variants/ml-experiments/00544-lag-light-demand-network-robust-same-slot-median-and-lag-dropout
for arm in R-MED R-DROP; do
  $D/run.sh fit --arm $arm && $D/run.sh assemble --arm $arm || exit 1
done
for arm in R-MED R-DROP; do $D/run.sh fit --arm $arm --stress --seeds 0 || exit 1; done
echo ALL DONE
