#!/bin/bash
# Stage 3: A8 (folded head; its own step 2), then A9 and A9-null (W-T columns: pretrain, stress, scored fits).
D="$(cd "$(dirname "$0")" && pwd)"
$D/run.sh step2 --arm A8 || exit 1
$D/run.sh fit --arm A8 --seeds 0 --stress || exit 1
$D/run.sh fit --arm A8 --seeds 0 1 || exit 1
$D/run.sh assemble --arm A8 || exit 1
for arm in A9 A9-null; do
  $D/run.sh pretrain-wt --arm $arm || exit 1
  $D/run.sh fit --arm $arm --seeds 0 --stress || exit 1
  $D/run.sh fit --arm $arm --seeds 0 1 || exit 1
  $D/run.sh assemble --arm $arm || exit 1
done
echo STAGE 3 DONE
