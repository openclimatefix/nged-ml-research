#!/bin/bash
D=/home/jack/dev/nged-ml-research/.claude/worktrees/aux-targets-F/ml-experiments/00543-lag-free-network-auxiliary-recent-level-and-reverse-flow-heads
for arm in ${ARMS:-F1 F3 F2}; do
  $D/run.sh fit --arm $arm && $D/run.sh assemble --arm $arm || exit 1
done
for arm in ${ARMS:-F1 F3 F2}; do $D/run.sh fit --arm $arm --stress --seeds 0 || exit 1; done
echo ALL DONE
