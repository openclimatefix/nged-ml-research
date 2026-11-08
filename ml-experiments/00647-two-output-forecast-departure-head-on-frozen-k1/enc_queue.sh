#!/bin/bash
# usage: enc_queue.sh; runs the stage-3 encoder fits one process at a time (nvidia-smi before each launch)
D="$(cd "$(dirname "$0")" && pwd)"
for job in "E1 2025-12 fc4" "E2 2025-12 fc4" "E3 2025-12 fc4" "E3p 2025-12 fc4"; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  $D/gpu.sh e00647_enc.py $job --seeds 0 1 2>&1 | grep -v mlflow || echo "FAIL $job"
done
echo QUEUEDONE
