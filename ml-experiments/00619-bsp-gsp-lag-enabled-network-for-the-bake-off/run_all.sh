#!/bin/bash
# Pretrain, step 2, fit, assemble for both groups; one GPU process at a time.
cd "$(cd "$(dirname "$0")" && pwd)"
for g in bsp gsp; do
  for cmd in pretrain-wt step2 fit assemble; do
    ./run.sh e00619_fit.py $cmd --group $g --seeds 0 1 || exit 1
  done
done
echo ALL DONE
