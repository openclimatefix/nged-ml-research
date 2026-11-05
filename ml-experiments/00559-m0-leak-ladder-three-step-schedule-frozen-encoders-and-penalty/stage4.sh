#!/bin/bash
# Stage 4: A11 (wind u/v columns, real and placebo): pretrain, seed-0 stress, scored fits, assemble.
D="$(cd "$(dirname "$0")" && pwd)"
for arm in A11 A11-null; do
  $D/run.sh pretrain-wt --arm $arm || exit 1
  $D/run.sh fit --arm $arm --seeds 0 --stress || exit 1
  $D/run.sh fit --arm $arm --seeds 0 1 || exit 1
  $D/run.sh assemble --arm $arm || exit 1
done
echo STAGE 4 DONE
