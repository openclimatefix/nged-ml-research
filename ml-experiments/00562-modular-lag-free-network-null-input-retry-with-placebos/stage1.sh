#!/bin/bash
# Stage 1: seed 0 of all arms, one GPU process at a time, then assemble each arm.
D="$(cd "$(dirname "$0")" && pwd)"
for arm in B1 B1p B2 B2p B3a B3ap B3b B3bp B4 B4p B4-oracle; do
  $D/run.sh fit --arm $arm --seeds 0 || exit 1
  $D/run.sh assemble --arm $arm --seeds 0 || exit 1
done
echo STAGE 1 DONE
