#!/bin/bash
# usage: c_inject.sh: after the G1 queue ends, inject banks S and X into the stored Candidate (experiment 612's weights)
D="$(cd "$(dirname "$0")" && pwd)"
export DATA_PATH_INTERNAL=${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}
W612=$DATA_PATH_INTERNAL/ml-experiments/00612-nra-combined-networks-three-class-roster-completion/P7_D50_e3
LOG=$DATA_PATH_INTERNAL/ml-experiments/00651-graph-layer-over-electrical-neighbours-in-the-demand-network-part-1/gpu_hours.log
while pgrep -f g1_queue.sh > /dev/null; do sleep 60; done
for b in S X; do
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
  t0=$(date +%s)
  flock /tmp/e648.gpu.lock "$D/gpu.sh" e00651_arm.py "P7+D50" inject --months 2025-10 2025-12 --seeds 0 1 --bank $b --weights-root $W612 || { echo "FAILED C inject $b" | tee -a "$LOG"; exit 5; }
  echo "C inject --bank $b $(( $(date +%s) - t0 ))" >> "$LOG"
done
