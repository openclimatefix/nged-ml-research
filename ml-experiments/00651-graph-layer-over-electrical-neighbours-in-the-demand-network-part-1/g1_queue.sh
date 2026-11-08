#!/bin/bash
# usage: g1_queue.sh [ARM_MODS ...]: run the G1 steps in order, one GPU process at a time.
# Each step waits until no other process uses the GPU, checks free disk (>= 5 GB) and the GPU-hour
# ceiling (5 h of process time, summed in gpu_hours.log), then runs and logs its seconds.
D="$(cd "$(dirname "$0")" && pwd)"
export DATA_PATH_INTERNAL=${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}
LOG=$DATA_PATH_INTERNAL/ml-experiments/00651-graph-layer-over-electrical-neighbours-in-the-demand-network-part-1/gpu_hours.log
CEILING_S=18000
mkdir -p "$(dirname "$LOG")"; touch "$LOG"
M="2025-10 2025-12"

step() {  # step ARM COMMAND [extra args]
  local arm=$1 cmd=$2; shift 2
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
  local free_gb; free_gb=$(df --output=avail -BG "$DATA_PATH_INTERNAL" | tail -1 | tr -dc 0-9)
  if [ "$free_gb" -lt 5 ]; then echo "STOP: only ${free_gb} GB free" | tee -a "$LOG"; exit 3; fi
  local used; used=$(awk '{s+=$NF} END {print s+0}' "$LOG")
  if [ "$used" -ge "$CEILING_S" ]; then echo "STOP: GPU ceiling reached (${used} s)" | tee -a "$LOG"; exit 4; fi
  local t0; t0=$(date +%s)
  flock /tmp/e648.gpu.lock "$D/gpu.sh" e00651_arm.py "$arm" "$cmd" --months $M --seeds 0 1 "$@" || { echo "FAILED $arm $cmd" | tee -a "$LOG"; exit 5; }
  echo "$arm $cmd $* $(( $(date +%s) - t0 ))" >> "$LOG"
}

for arm in "$@"; do
  step "$arm" fit
  for b in M S X; do step "$arm" inject --bank $b; done
done
