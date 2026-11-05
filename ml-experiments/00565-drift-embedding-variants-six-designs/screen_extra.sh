#!/bin/bash
# Seeds 2 to 4 of C-M0 and D5 (coarse), to test whether D5's near-pass holds. One GPU process.
D="$(cd "$(dirname "$0")" && pwd)"
$D/run.sh pretrain-wt --arm D5 --seeds 2 3 4 || exit 1
for arm in C-M0 D5; do
  $D/run.sh fit --arm $arm --seeds 2 3 4 || exit 1
  $D/run.sh assemble --arm $arm --seeds 2 3 4 || exit 1
done
echo EXTRA DONE
