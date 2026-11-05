#!/bin/bash
# Extra fold months covering the telemetry-verified transfers of experiment 580 inside the ENS-summary period.
# Experiment 582's worker owns the v3+D5 fits for these months (reused, never refitted here); this script fits
# X3-med and forecasts both arms with the verified events removed from the lags (pass 100). Resumable.
set -e
D="$(cd "$(dirname "$0")" && pwd)"
M="2026-05"
W582="$DATA_PATH_INTERNAL/ml-experiments/00582-nra-tails-on-verified-load-shift-events/work"
$D/run.sh x3med pretrain --months $M
$D/run.sh x3med fit --months $M
$D/run.sh x3med assemble --months $M
$D/run.sh x3med inject --months $M --passes 100
for m in $M; do
  while [ ! -f "$W582/models/V3D5/${m}_d14_s1.pt" ]; do sleep 60; done
done
$D/run.sh v3d5 inject --months $M --passes 100
echo EXTRA DONE
