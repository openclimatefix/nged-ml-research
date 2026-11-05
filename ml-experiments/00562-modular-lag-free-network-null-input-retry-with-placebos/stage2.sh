#!/bin/bash
# Stage 2: seed 1 for the arms that passed or sat near the gate at seed 0, real and placebo together.
D="$(cd "$(dirname "$0")" && pwd)"
for arm in B1 B1p B2 B2p B3b B3bp; do
  $D/run.sh fit --arm $arm --seeds 1 || exit 1
  $D/run.sh assemble --arm $arm --seeds 1 || exit 1
done
echo STAGE 2 DONE
