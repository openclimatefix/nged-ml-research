#!/bin/bash
# usage: bisect.sh TAG SKIP: forecast the stored seed-2 weights of 2026-06 with some redirects switched off, to find which one moves the forecast
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00620-bakeoff-final-test-demand-networks-at-primaries
S=$DATA_PATH_INTERNAL/ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/arms/k1_d5/work
mkdir -p $R/arms/$1/work/models/P5
cp $S/models/P5/2026-06_d0_s2.pt $R/arms/$1/work/models/P5/
export E00620_PHASE=predict CHAIN620=$R/chains/k1/work E00620_SKIP=$2
$D/gpu.sh e00620_k1.py $1 fit --months 2026-06 --seeds 2 > $R/logs/$1.log 2>&1
