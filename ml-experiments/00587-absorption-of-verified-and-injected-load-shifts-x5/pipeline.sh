#!/bin/bash
# the three arms one after another (one GPU process at a time, 8 GB host cap each)
D="$(cd "$(dirname "$0")" && pwd)"
L="$DATA_PATH_INTERNAL/ml-experiments/00587-absorption-of-verified-and-injected-load-shifts-x5/logs"
mkdir -p "$L"
"$D/run.sh" 565 --arm M0 > "$L/M0.log" 2>&1
"$D/run.sh" 565 --arm D5 > "$L/D5.log" 2>&1
"$D/run.sh" 574 --arm P5 > "$L/P5.log" 2>&1
echo finished > "$L/pipeline.done"
