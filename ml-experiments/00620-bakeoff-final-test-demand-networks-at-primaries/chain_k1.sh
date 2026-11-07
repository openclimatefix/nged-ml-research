#!/bin/bash
# usage: chain_k1.sh: pre-training, step 2, and the single fit of K1 on the final fold (train phase only: the held-out power is hidden)
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00620-bakeoff-final-test-demand-networks-at-primaries
export E00620_PHASE=train CHAIN620=$R/chains/k1/work
set -e
for c in pretrain step2 fit; do
  $D/gpu.sh e00620_k1.py k1 $c --months 2026-07 --seeds 0 1 2 3 4 > $R/logs/k1_$c.log 2>&1
done
