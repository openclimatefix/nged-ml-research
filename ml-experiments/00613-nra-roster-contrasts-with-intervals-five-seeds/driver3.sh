#!/bin/bash
# usage: driver3.sh: queue the K3 O2 fits and the matching K3 D5 fits on seeds 5 to 7
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds
M="2025-10 2025-12 2026-01 2026-03 2026-06"
$D/queue.sh $R/logs/k3_o2.log $D/gpu.sh e00613_free_o2.py k3_o2 --months $M --seeds 5 6 7 &
sleep 10
$D/queue.sh $R/logs/k3_d5b.log $D/gpu.sh e00613_free.py k3_d5b --months $M --seeds 5 6 7 &
wait
