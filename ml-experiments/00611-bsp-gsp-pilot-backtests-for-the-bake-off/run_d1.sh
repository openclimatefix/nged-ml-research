#!/bin/bash
# Design 1: pretrain, fit, assemble for both groups; one GPU process at a time.
cd "$(cd "$(dirname "$0")" && pwd)"
for g in bsp gsp; do
  ./run.sh e00611_fit.py pretrain --group $g --seeds 0 1 || exit 1
  ./run.sh e00611_fit.py fit --group $g --seeds 0 1 || exit 1
  ./run.sh e00611_fit.py assemble --group $g --seeds 0 1 || exit 1
done
echo D1 DONE
