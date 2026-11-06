#!/bin/bash
# the two arms one after another (one GPU process at a time, 8 GB host cap each)
# stops at the first failed arm
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
D="$(cd "$(dirname "$0")" && pwd)"
L="$DATA_PATH_INTERNAL/ml-experiments/00595-nra-target-side-absorption-for-p5-lag-encoder-with-synthetic-shifts/logs"
mkdir -p "$L"
"$D/run.sh" C1 > "$L/C1.log" 2>&1 && "$D/run.sh" A1 > "$L/A1.log" 2>&1 && echo finished > "$L/pipeline.done"
