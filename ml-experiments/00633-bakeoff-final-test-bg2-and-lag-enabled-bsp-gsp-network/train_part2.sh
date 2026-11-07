#!/bin/bash
# Part 2 train phase: pretrain-wt, step2, fit for each group; window hidden. One GPU process at a time.
D="$(cd "$(dirname "$0")" && pwd)"
export E00633_PHASE=train
for g in bsp gsp; do
  for cmd in pretrain-wt step2 fit; do
    $D/gpu.sh e00633_node_fit.py $cmd --group $g --months 2026-07 --seeds 0 1 || exit 1
  done
done
echo TRAIN DONE
