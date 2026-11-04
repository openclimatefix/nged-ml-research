#!/bin/bash
# Stage 6: A11c (total cloud cover, real and placebo), limited history: folds from 2025-12 only.
D="$(cd "$(dirname "$0")" && pwd)"
M="2025-12 2026-01 2026-03 2026-06"
for arm in A11c A11c-null; do
  $D/run.sh pretrain-wt --arm $arm --months $M || exit 1
  $D/run.sh fit --arm $arm --seeds 0 --stress --months $M || exit 1
  $D/run.sh fit --arm $arm --seeds 0 1 --months $M || exit 1
done
echo STAGE 6 DONE
