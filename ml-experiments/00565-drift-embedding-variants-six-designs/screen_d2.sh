#!/bin/bash
# Design 2 (slow-trend summary input) and its placebo, coarse cadence, seeds 0 and 1, injected shift on seed 0. Waits for screen_extra.sh.
D="$(cd "$(dirname "$0")" && pwd)"
while pgrep -f "00565-drift-embedding-variants-six-designs/screen_extra" >/dev/null; do sleep 30; done
for arm in D2 D2-null; do
  $D/run.sh pretrain-wt --arm $arm --seeds 0 1 || exit 1
  $D/run.sh fit --arm $arm --seeds 0 1 || exit 1
  $D/run.sh fit --arm $arm --seeds 0 --stress || exit 1
  $D/run.sh assemble --arm $arm --seeds 0 1 || exit 1
  echo "SCREEN $arm DONE"
done
echo D2 ALL DONE
