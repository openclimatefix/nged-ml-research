#!/bin/bash
# Stage 1: main coarse runs (cut day 0), seed 0, one GPU process at a time; M0 also seed 1 for the noise floor.
D="$(cd "$(dirname "$0")" && pwd)"
$D/run.sh fit --arm M0 --seeds 0 1 --cut-day 0 || exit 1
$D/run.sh assemble --arm M0 --seeds 0 1 || exit 1
for arm in B3a B3ap B1 B1p B3c B3b B3bp B4 B4p B2 B2p; do
  $D/run.sh pretrain --arm $arm --seeds 0 || exit 1
  $D/run.sh fit --arm $arm --seeds 0 --cut-day 0 || exit 1
  $D/run.sh assemble --arm $arm --seeds 0 || exit 1
  echo ARM DONE $arm
done
echo STAGE 1 DONE
