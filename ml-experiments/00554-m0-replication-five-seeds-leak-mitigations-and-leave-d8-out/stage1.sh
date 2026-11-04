#!/bin/bash
# Stage 1: the 5-seed controls (seeds 2-4 here; seeds 0-1 are 511's and 545's), then assemble all 5 seeds.
D="$(cd "$(dirname "$0")" && pwd)"
for arm in L-1d M0lr M0; do
  $D/run.sh fit --arm $arm --seeds 2 3 4 || exit 1
  $D/run.sh assemble --arm $arm || exit 1
done
echo STAGE 1 DONE
