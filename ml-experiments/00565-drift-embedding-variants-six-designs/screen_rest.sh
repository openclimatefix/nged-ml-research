#!/bin/bash
# After the per-event refits: hybrid H (D5 + A10) on seeds 0 and 1 with the injected shift, then the
# seeds 5 to 9 confirmation of D5 against coarse M0. One GPU process at a time.
D="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$D/../.." && pwd)"
while pgrep -f "00565-drift-embedding-variants-six-designs/screen_(d|events)" >/dev/null; do sleep 30; done
$D/run.sh pretrain-wt --arm H --seeds 0 1 || exit 1
$D/run.sh fit --arm H --seeds 0 1 || exit 1
$D/run.sh fit --arm H --seeds 0 --stress || exit 1
$D/run.sh assemble --arm H --seeds 0 1 || exit 1
echo HYBRID DONE
$R/ml-experiments/00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out/run.sh pretrain --arm M0 --seeds 5 6 7 8 9 || exit 1
$D/run.sh pretrain-wt --arm D5 --seeds 5 6 7 8 9 || exit 1
for arm in C-M0 D5; do
  $D/run.sh fit --arm $arm --seeds 5 6 7 8 9 || exit 1
  $D/run.sh assemble --arm $arm --seeds 5 6 7 8 9 || exit 1
done
echo CONFIRM DONE
