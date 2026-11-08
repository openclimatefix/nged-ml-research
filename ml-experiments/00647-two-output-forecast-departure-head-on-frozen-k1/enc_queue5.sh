#!/bin/bash
# usage: enc_queue5.sh; stage 4: E2 and E1 leave-one-month-out on all five months, K1 seeds 0 to 4 (one GPU process at a time)
D="$(cd "$(dirname "$0")" && pwd)"
for arm in E2 E1; do
  for m in 2025-10 2025-12 2026-01 2026-03 2026-06; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    $D/gpu.sh e00647_enc.py $arm $m lomo --seeds 0 1 2 3 4 2>&1 | grep -v mlflow || echo "FAIL $arm $m"
  done
done
echo QUEUEDONE
