#!/bin/bash
# the three arms one after another (one GPU process at a time, 8 GB host cap each); stops at the first failure
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
D="$(cd "$(dirname "$0")" && pwd)"
L="$DATA_PATH_INTERNAL/ml-experiments/00599-nra-refit-embargo-and-down-weighting-of-the-newest-rows-against-load-shift-absorption/logs"
mkdir -p "$L"
"$D/run.sh" E1 > "$L/E1.log" 2>&1 && "$D/run.sh" E2w > "$L/E2w.log" 2>&1 && "$D/run.sh" E3 > "$L/E3.log" 2>&1 && echo finished > "$L/pipeline.done"
