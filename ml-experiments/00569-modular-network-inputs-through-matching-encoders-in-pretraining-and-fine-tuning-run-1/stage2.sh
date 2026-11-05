#!/bin/bash
# Stage 2: injected 7-day shift leak check at cut day 14 (control fit, then the shifted fits), seed 0.
D="$(cd "$(dirname "$0")" && pwd)"
for arm in M0 B1 B1p B3a B3ap; do
  $D/run.sh fit --arm $arm --seeds 0 --cut-day 14 || exit 1
  $D/run.sh fit --arm $arm --seeds 0 --cut-day 14 --stress || exit 1
  echo ARM DONE $arm
done
echo STAGE 2 DONE
